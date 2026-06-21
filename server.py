from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import time
import asyncio
import traceback
import base64
from typing import List, Optional
from PIL import Image
import io

from models import GenerateRequest, EncounterResult, ClinicalAlert, SOAPNote
from medgemma_engine import (
    run_clinical_pipeline,
    run_clinical_pipeline_streaming,
    run_clinical_pipeline_async,
    analyze_medical_image,
    pipe as mg_pipe,
    _executor,
)
from transcriber import transcribe_audio, whisper_model as wh
from database import save_encounter, get_encounters, delete_encounter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(SCRIPT_DIR, "uploads")

app = FastAPI(title="MediScribe", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

os.makedirs(os.path.join(SCRIPT_DIR, "static"), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
async def root():
    return FileResponse(os.path.join(SCRIPT_DIR, "static", "index.html"))


@app.get("/health")
async def health():
    from medgemma_engine import pipe as mg
    from transcriber import whisper_model as wh
    return {
        "status": "ok",
        "medgemma_loaded": mg is not None,
        "whisper_loaded": wh is not None,
    }


@app.post("/api/transcribe")
async def api_transcribe(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(400, "Empty audio file")
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, transcribe_audio, audio_bytes)
        return result
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")


@app.post("/api/analyze-image")
async def api_analyze_image(image: UploadFile = File(...)):
    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(400, "Empty image file")
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        safe_name = f"{int(time.time())}_{image.filename}"
        save_path = os.path.join(UPLOAD_DIR, safe_name)
        pil_image.save(save_path)
        print(f"[Server] Analyzing image: {image.filename}")
        loop = asyncio.get_event_loop()
        analysis = await loop.run_in_executor(_executor, analyze_medical_image, pil_image, safe_name)
        return analysis.model_dump()
    except Exception as e:
        print(f"[Server] Image analysis FAILED: {e}")
        traceback.print_exc()
        raise HTTPException(500, f"Image analysis failed: {e}")


@app.post("/api/generate")
async def api_generate(
    transcript: str = Form(...),
    patient_name: str = Form(""),
    patient_age: Optional[int] = Form(None),
    patient_sex: Optional[str] = Form(None),
    images: List[UploadFile] = File(default=[]),
):
    if not transcript.strip():
        raise HTTPException(400, "Empty transcript")
    t0 = time.time()
    try:
        print(f"[Server] /api/generate - patient={patient_name}, transcript={len(transcript)}, images={len(images)}")
        loop = asyncio.get_event_loop()
        encounter = await loop.run_in_executor(_executor, run_clinical_pipeline, transcript, patient_age, patient_sex)
        encounter.patient_name = patient_name

        for img_file in images:
            try:
                img_bytes = await img_file.read()
                if len(img_bytes) == 0:
                    continue
                pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                safe_name = f"{int(time.time())}_{img_file.filename}"
                save_path = os.path.join(UPLOAD_DIR, safe_name)
                pil_image.save(save_path)
                print(f"[Server] Analyzing uploaded image: {img_file.filename}")
                analysis = await loop.run_in_executor(_executor, analyze_medical_image, pil_image, safe_name)
                encounter.image_analyses.append(analysis)
            except Exception as e:
                print(f"[Server] Image {img_file.filename} analysis failed: {e}")
                traceback.print_exc()

        print(f"[Server] Pipeline done in {encounter.processing_time_ms}ms")
    except Exception as e:
        print(f"[Server] Pipeline EXCEPTION: {e}")
        traceback.print_exc()
        encounter = EncounterResult(transcript=transcript, patient_name=patient_name)
        encounter.soap_note = SOAPNote(
            subjective=transcript.strip(),
            objective="Pipeline error occurred.",
            assessment=str(e),
            plan="Restart and retry.",
        )
        encounter.processing_time_ms = round((time.time() - t0) * 1000, 1)
    save_encounter(encounter.model_dump())
    return encounter.model_dump()


@app.post("/api/generate-stream")
async def api_generate_stream(
    transcript: str = Form(...),
    patient_name: str = Form(""),
    patient_age: Optional[int] = Form(None),
    patient_sex: Optional[str] = Form(None),
    images: List[UploadFile] = File(default=[]),
):
    if not transcript.strip():
        raise HTTPException(400, "Empty transcript")

    image_data = []
    for img_file in images:
        try:
            img_bytes = await img_file.read()
            if len(img_bytes) > 0:
                image_data.append((img_file.filename, img_bytes))
        except Exception as e:
            print(f"[Server] Failed to read image {img_file.filename}: {e}")

    async def event_generator():
        t0 = time.time()
        encounter = None
        queue = asyncio.Queue()

        try:
            print(f"[Server] /api/generate-stream - patient={patient_name}, "
                  f"transcript={len(transcript)}, images={len(image_data)}")

            pipeline_task = asyncio.create_task(
                run_clinical_pipeline_async(transcript, queue, patient_age, patient_sex)
            )

            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=120.0)
                except asyncio.TimeoutError:
                    break

                if event.get("status") == "complete" and "result" in event:
                    encounter = event["result"]
                    break
                else:
                    sse_data = json.dumps(event)
                    yield f"data: {sse_data}\n\n"

            await pipeline_task

            if encounter is None:
                encounter = EncounterResult(transcript=transcript)

            encounter.patient_name = patient_name

            total_with_images = 7 + len(image_data)
            for idx, (filename, img_bytes) in enumerate(image_data, start=1):
                step_num = 7 + idx
                yield f"data: {json.dumps({'step': step_num, 'total': total_with_images, 'status': 'running', 'label': f'Analyzing image: {filename}...'})}\n\n"
                img_t0 = time.time()
                try:
                    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    safe_name = f"{int(time.time())}_{filename}"
                    save_path = os.path.join(UPLOAD_DIR, safe_name)
                    pil_image.save(save_path)
                    loop = asyncio.get_event_loop()
                    analysis = await loop.run_in_executor(_executor, analyze_medical_image, pil_image, safe_name)
                    encounter.image_analyses.append(analysis)
                    img_ms = round((time.time() - img_t0) * 1000, 1)
                    yield f"data: {json.dumps({'step': step_num, 'total': total_with_images, 'status': 'done', 'label': f'Image analyzed: {filename}', 'time_ms': img_ms})}\n\n"
                except Exception as e:
                    img_ms = round((time.time() - img_t0) * 1000, 1)
                    print(f"[Server] Image {filename} analysis failed: {e}")
                    traceback.print_exc()
                    yield f"data: {json.dumps({'step': step_num, 'total': total_with_images, 'status': 'error', 'label': f'Image failed: {filename}', 'time_ms': img_ms})}\n\n"

            encounter.processing_time_ms = round((time.time() - t0) * 1000, 1)

            save_encounter(encounter.model_dump())

            final_event = {
                "step": total_with_images if image_data else 7,
                "total": total_with_images if image_data else 7,
                "status": "complete",
                "label": "Pipeline complete",
                "result": encounter.model_dump(),
            }
            yield f"data: {json.dumps(final_event)}\n\n"

        except Exception as e:
            print(f"[Server] Stream EXCEPTION: {e}")
            traceback.print_exc()
            if encounter is None:
                encounter = EncounterResult(transcript=transcript, patient_name=patient_name)
                encounter.soap_note = SOAPNote(
                    subjective=transcript.strip(),
                    objective="Pipeline error occurred.",
                    assessment=str(e),
                    plan="Restart and retry.",
                )
            encounter.processing_time_ms = round((time.time() - t0) * 1000, 1)
            save_encounter(encounter.model_dump())
            error_event = {
                "step": 7, "total": 7, "status": "complete",
                "label": "Pipeline complete (with errors)",
                "result": encounter.model_dump(),
            }
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/full-pipeline")
async def api_full_pipeline(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(400, "Empty audio file")
    try:
        loop = asyncio.get_event_loop()
        transcription = await loop.run_in_executor(_executor, transcribe_audio, audio_bytes)
        encounter = await loop.run_in_executor(_executor, run_clinical_pipeline, transcription["text"])
        data = encounter.model_dump()
        data["transcription"] = transcription
        save_encounter(data)
        return data
    except Exception as e:
        raise HTTPException(500, f"Pipeline failed: {e}")


@app.get("/api/encounters")
async def api_encounters():
    return get_encounters()


@app.delete("/api/encounters/{encounter_id}")
async def api_delete(encounter_id: str):
    delete_encounter(encounter_id)
    return {"status": "deleted"}


app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/static", StaticFiles(directory=os.path.join(SCRIPT_DIR, "static")), name="static")
