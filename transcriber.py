from faster_whisper import WhisperModel
import tempfile
import os
import time
import torch

whisper_model = None


def load_whisper(model_size="base"):
    global whisper_model
    if whisper_model is not None:
        print("[Whisper] Already loaded.")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ct = "float16" if device == "cuda" else "int8"
    print(f"[Whisper] Loading {model_size} on {device} ({ct}) ...")
    t0 = time.time()
    try:
        whisper_model = WhisperModel(model_size, device=device, compute_type=ct)
    except Exception:
        print("[Whisper] GPU failed, falling back to CPU ...")
        whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
    print(f"[Whisper] Ready in {time.time()-t0:.1f}s")


def transcribe_audio(audio_bytes, language="en"):
    if whisper_model is None:
        raise RuntimeError("Whisper not loaded. Run Cell 7 first.")
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name
    try:
        t0 = time.time()
        segments, info = whisper_model.transcribe(tmp, language=language, beam_size=5,
                                                   vad_filter=True)
        text = ""
        segs = []
        for s in segments:
            text += s.text + " "
            segs.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})
        return {
            "text": text.strip(),
            "segments": segs,
            "language": info.language,
            "duration": info.duration,
            "time_ms": round((time.time() - t0) * 1000, 1),
        }
    finally:
        os.unlink(tmp)
