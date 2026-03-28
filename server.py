from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import time
import traceback
import io
from typing import List, Optional
from PIL import Image
from models import EncounterResult, ClinicalAlert, SOAPNote
from medgemma_engine import run_clinical_pipeline, analyze_medical_image, pipe as mg_pipe
from transcriber import transcribe_audio, whisper_model as wh, SUPPORTED_LANGUAGES
from database import save_encounter, get_encounters, search_encounters, delete_encounter
from config import MAX_UPLOAD_BYTES, ALLOWED_IMAGE_TYPES
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(SCRIPT_DIR, "uploads")
app = FastAPI(title="MediScribe", version="4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
os.makedirs(os.path.join(SCRIPT_DIR, "static"), exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
@app.get("/")
async def root():
    return FileResponse(os.path.join(SCRIPT_DIR, "static", "index.html"))
@app.get("/health")
async def health():
    from medgemma_engine import pipe as mg
    from transcriber import whisper_model as wh_mod
    return {
        "status": "ok",
        "medgemma_loaded": mg is not None,
        "whisper_loaded": wh_mod is not None,
    }
@app.get("/api/languages")
async def api_languages():
    return SUPPORTED_LANGUAGES
@app.post("/api/transcribe")
async def api_transcribe(
    audio: UploadFile = File(...),
    language: str = Form("en"),
):
    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(400, "Empty audio file")
    try:
        return transcribe_audio(audio_bytes, language=language)
    except Exception as e:
        raise HTTPException(500, f"Transcription failed: {e}")
def _validate_image(image_bytes: bytes, filename: str):
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Image '{filename}' exceeds {MAX_UPLOAD_BYTES // (1024*1024)}MB limit")
    try:
        img = Image.open(io.BytesIO(image_bytes))
        fmt = (img.format or "").lower()
        allowed_formats = {"jpeg", "png", "tiff", "jpg"}
        if fmt not in allowed_formats:
            raise HTTPException(415, f"Unsupported image format '{fmt}'. Allowed: JPEG, PNG, TIFF")
        return img.convert("RGB")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Invalid image file '{filename}': {e}")
@app.post("/api/analyze-image")
async def api_analyze_image(image: UploadFile = File(...)):
    image_bytes = await image.read()
    if len(image_bytes) == 0:
        raise HTTPException(400, "Empty image file")
    pil_image = _validate_image(image_bytes, image.filename or "image")
    try:
        safe_name = f"{int(time.time())}_{image.filename}"
        save_path = os.path.join(UPLOAD_DIR, safe_name)
        pil_image.save(save_path)
        print(f"[Server] Analyzing image: {image.filename}")
        analysis = analyze_medical_image(pil_image, filename=safe_name)
        return analysis.model_dump()
    except HTTPException:
        raise
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
    language: str = Form("en"),
    images: List[UploadFile] = File(default=[]),
):
    if not transcript.strip():
        raise HTTPException(400, "Empty transcript")
    t0 = time.time()
    try:
        print(f"[Server] /api/generate patient={patient_name} lang={language} transcript={len(transcript)} images={len(images)}")
        encounter = run_clinical_pipeline(transcript, patient_age, patient_sex)
        encounter.patient_name = patient_name
        for img_file in images:
            try:
                img_bytes = await img_file.read()
                if len(img_bytes) == 0:
                    continue
                pil_image = _validate_image(img_bytes, img_file.filename or "img")
                safe_name = f"{int(time.time())}_{img_file.filename}"
                pil_image.save(os.path.join(UPLOAD_DIR, safe_name))
                print(f"[Server] Analyzing image: {img_file.filename}")
                analysis = analyze_medical_image(pil_image, filename=safe_name)
                encounter.image_analyses.append(analysis)
            except HTTPException as he:
                print(f"[Server] Image rejected: {he.detail}")
            except Exception as e:
                print(f"[Server] Image {img_file.filename} failed: {e}")
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
@app.post("/api/full-pipeline")
async def api_full_pipeline(
    audio: UploadFile = File(...),
    language: str = Form("en"),
):
    audio_bytes = await audio.read()
    if len(audio_bytes) == 0:
        raise HTTPException(400, "Empty audio file")
    try:
        transcription = transcribe_audio(audio_bytes, language=language)
        encounter = run_clinical_pipeline(transcription["text"])
        data = encounter.model_dump()
        data["transcription"] = transcription
        save_encounter(data)
        return data
    except Exception as e:
        raise HTTPException(500, f"Pipeline failed: {e}")
@app.get("/api/encounters")
async def api_encounters(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
):
    return get_encounters(page=page, limit=limit)
@app.get("/api/encounters/search")
async def api_search(
    q: str = Query(""),
    patient: str = Query(""),
    date_from: str = Query(""),
    date_to: str = Query(""),
):
    return search_encounters(q=q, patient=patient, date_from=date_from, date_to=date_to)
@app.delete("/api/encounters/{encounter_id}")
async def api_delete(encounter_id: str):
    delete_encounter(encounter_id)
    return {"status": "deleted"}
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/static", StaticFiles(directory=os.path.join(SCRIPT_DIR, "static")), name="static")
