import os
from dotenv import load_dotenv
_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_dir, ".env"))
PORT                     = int(os.getenv("PORT", 7860))
WHISPER_MODEL            = os.getenv("WHISPER_MODEL", "small")
MEDGEMMA_MODEL           = os.getenv("MEDGEMMA_MODEL", "google/medgemma-4b-it")
SARVAM_API_KEY           = os.getenv("SARVAM_API_KEY", "")
DB_PATH                  = os.getenv("DB_PATH", os.path.join(_dir, "mediscribe.db"))
MAX_UPLOAD_BYTES         = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
ALLOWED_IMAGE_TYPES      = {"image/jpeg", "image/png", "image/tiff", "image/jpg"}
ICD10_CONFIDENCE_THRESHOLD = float(os.getenv("ICD10_CONFIDENCE_THRESHOLD", "0.40"))
