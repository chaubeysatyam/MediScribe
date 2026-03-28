import subprocess, sys
def run(cmd):
    print(f">> {cmd}")
    subprocess.check_call(cmd, shell=True)
run(f"{sys.executable} -m pip install -q torch>=2.6 torchvision --index-url https://download.pytorch.org/whl/cu121")
run(f"{sys.executable} -m pip install -q transformers>=4.50.0 accelerate bitsandbytes huggingface_hub")
run(f"{sys.executable} -m pip install -q faster-whisper uvicorn fastapi python-multipart pydantic")
import platform
if platform.system() == "Linux":
    run("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared")
    run("chmod +x /usr/local/bin/cloudflared")
print()
print("[Cell 1] DONE. Now RESTART Python/runtime, then run Cell 2.")
