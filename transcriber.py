from faster_whisper import WhisperModel
from typing import Optional
import tempfile
import os
import time
import torch
from config import WHISPER_MODEL, SARVAM_API_KEY

whisper_model = None
_sarvam_client = None

WHISPER_TO_SARVAM = {
    "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN",
    "kn": "kn-IN", "ml": "ml-IN", "gu": "gu-IN", "mr": "mr-IN",
    "pa": "pa-IN", "ur": "ur-IN", "od": "od-IN",
}

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi (हिन्दी)",
    "bn": "Bengali (বাংলা)",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "ml": "Malayalam (മലയാളം)",
    "gu": "Gujarati (ગુજરાતી)",
    "mr": "Marathi (मराठी)",
    "pa": "Punjabi (ਪੰਜਾਬੀ)",
    "ur": "Urdu (اردو)",
}


def _get_sarvam():
    global _sarvam_client
    if _sarvam_client is None and SARVAM_API_KEY:
        try:
            from sarvamai import SarvamAI
            _sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)
            print("[Sarvam] Client initialised.")
        except Exception as e:
            print(f"[Sarvam] Init error: {e}")
    return _sarvam_client


def load_whisper(model_size=None):
    global whisper_model
    if whisper_model is not None:
        print("[Whisper] Already loaded.")
        return
    size = model_size or WHISPER_MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ct = "float16" if device == "cuda" else "int8"
    print(f"[Whisper] Loading '{size}' on {device} ({ct}) ...")
    t0 = time.time()
    try:
        whisper_model = WhisperModel(size, device=device, compute_type=ct)
    except Exception:
        print("[Whisper] GPU failed, using CPU ...")
        whisper_model = WhisperModel(size, device="cpu", compute_type="int8")
    print(f"[Whisper] Ready in {time.time()-t0:.1f}s")


def _sarvam_translate(text, source_lang):
    client = _get_sarvam()
    if not client:
        return text, False
    sarvam_src = WHISPER_TO_SARVAM.get(source_lang, "auto")

    def _chunk_translate(chunk):
        try:
            resp = client.text.translate(
                input=chunk,
                source_language_code=sarvam_src,
                target_language_code="en-IN",
            )
            if hasattr(resp, "translated_text"):
                return resp.translated_text or chunk
            if isinstance(resp, dict):
                return resp.get("translated_text", chunk)
            return str(resp) or chunk
        except Exception as e:
            print(f"[Sarvam] Chunk error: {e}")
            return chunk

    if len(text) <= 900:
        translated = _chunk_translate(text)
    else:
        words, chunks, current = text.split(), [], ""
        for w in words:
            if len(current) + len(w) + 1 > 900:
                chunks.append(current.strip())
                current = w
            else:
                current += " " + w
        if current.strip():
            chunks.append(current.strip())
        translated = " ".join([_chunk_translate(c) for c in chunks])

    print(f"[Sarvam] {source_lang}->en: {translated[:80]}...")
    return translated, True


def transcribe_audio(audio_bytes, language="en"):
    if whisper_model is None:
        raise RuntimeError("Whisper not loaded.")

    suffix = ".webm"
    if len(audio_bytes) >= 8:
        if audio_bytes[4:8] == b"ftyp":
            suffix = ".mp4"
        elif audio_bytes[0:4] == b"OggS":
            suffix = ".ogg"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp = f.name

    try:
        t0 = time.time()
        whisper_lang = None if language == "auto" else language
        segments, info = whisper_model.transcribe(
            tmp, language=whisper_lang, beam_size=5, vad_filter=True
        )
        text = ""
        segs = []
        for s in segments:
            text += s.text + " "
            segs.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()})
        text = text.strip()
        detected_lang = info.language or language
        elapsed_ms = round((time.time() - t0) * 1000, 1)

        result = {
            "text": text,
            "segments": segs,
            "language": detected_lang,
            "duration": info.duration,
            "time_ms": elapsed_ms,
            "translated": False,
        }

        if not detected_lang.startswith("en") and text:
            print(f"[Transcriber] Non-English ({detected_lang}), translating via Sarvam ...")
            translated_text, was_translated = _sarvam_translate(text, detected_lang)
            if was_translated and translated_text != text:
                result["original_text"] = text
                result["text"] = translated_text
                result["translated"] = True

        return result
    finally:
        os.unlink(tmp)
