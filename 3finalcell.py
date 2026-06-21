# Cell 8 - Load Models + Start Server (Local PC)
# Run cells 1-5 first, then run THIS cell.

import sys, os, time, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Step 1: Load MedGemma ──
print("=" * 60)
print("  STEP 1/3: Loading MedGemma via pipeline ...")
print("=" * 60)
import medgemma_engine
if medgemma_engine.pipe is not None:
    print("[Cell 8] MedGemma already loaded. Skipping.")
else:
    medgemma_engine.load_medgemma()
    print(f"[Cell 8] MedGemma loaded! pipe = {type(medgemma_engine.pipe)}")

# ── Step 2: Load Whisper ──
print()
print("=" * 60)
print("  STEP 2/3: Loading Whisper ...")
print("=" * 60)
from transcriber import load_whisper, whisper_model
if whisper_model is not None:
    print("[Cell 8] Whisper already loaded. Skipping.")
else:
    load_whisper("base")
    print("[Cell 8] Whisper loaded!")

# ── Step 3: Start Server ──
print()
print("=" * 60)
print("  STEP 3/3: Starting server ...")
print("=" * 60)

import uvicorn
from server import app

PORT = 7860

def _run():
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

t = threading.Thread(target=_run, daemon=True)
t.start()
print(f"[Cell 8] Server starting on port {PORT} ...")
time.sleep(3)

# Health check
import urllib.request
try:
    r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
    print(f"[Cell 8] Health: {r.read().decode()}")
except Exception as e:
    print(f"[Cell 8] Health check failed: {e}")

print()
print("=" * 60)
print(f"  OPEN IN BROWSER:  http://127.0.0.1:{PORT}")
print("=" * 60)
print()
print("[Cell 8] Server is running. Press Ctrl+C to stop.")
print()

# ── Keep alive loop ──
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
    print("\n[Cell 8] Server stopped by user.")
