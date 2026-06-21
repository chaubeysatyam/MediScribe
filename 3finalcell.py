import sys, os, time, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  STEP 1/3: Loading MedGemma + Whisper in PARALLEL on GPU")
print("=" * 60)

import medgemma_engine
from transcriber import load_whisper, whisper_model

load_errors = []

def _load_medgemma():
    try:
        if medgemma_engine.pipe is None:
            medgemma_engine.load_medgemma()
            print(f"[Loader] MedGemma loaded on GPU! pipe = {type(medgemma_engine.pipe)}")
        else:
            print("[Loader] MedGemma already loaded.")
    except Exception as e:
        load_errors.append(f"MedGemma: {e}")
        print(f"[Loader] MedGemma FAILED: {e}")

def _load_whisper():
    try:
        from transcriber import whisper_model as wm
        if wm is None:
            load_whisper("base")
            print("[Loader] Whisper loaded on GPU!")
        else:
            print("[Loader] Whisper already loaded.")
    except Exception as e:
        load_errors.append(f"Whisper: {e}")
        print(f"[Loader] Whisper FAILED: {e}")

t0 = time.time()
t_mg = threading.Thread(target=_load_medgemma)
t_wh = threading.Thread(target=_load_whisper)
t_mg.start()
t_wh.start()
t_mg.join()
t_wh.join()
print(f"[Loader] Both models loaded in {time.time()-t0:.1f}s")

if load_errors:
    for err in load_errors:
        print(f"[Loader] ERROR: {err}")

print()
print("=" * 60)
print("  STEP 2/3: Starting server ...")
print("=" * 60)

import uvicorn
from server import app

PORT = 7860

def _run():
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

t = threading.Thread(target=_run, daemon=True)
t.start()
print(f"[Startup] Server starting on port {PORT} ...")
time.sleep(3)

import urllib.request
try:
    r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
    print(f"[Startup] Health: {r.read().decode()}")
except Exception as e:
    print(f"[Startup] Health check failed: {e}")

print()
print("=" * 60)
print(f"  OPEN IN BROWSER:  http://127.0.0.1:{PORT}")
print("=" * 60)
print()
print("[Startup] Server is running. Press Ctrl+C to stop.")
print()

try:
    counter = 0
    while True:
        time.sleep(60)
        counter += 1
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
            status = r.read().decode()
            print(f"[Keep-alive {counter}m] Server OK - {status}")
        except Exception as e:
            print(f"[Keep-alive {counter}m] WARNING: {e}")
except KeyboardInterrupt:
    print("\n[Startup] Server stopped by user.")
