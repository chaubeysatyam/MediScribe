import os
token = os.environ.get("HF_TOKEN")
if not token:
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    print("[Auth] Got token from .env file")
                    break
if token:
    os.environ["HF_TOKEN"] = token
    from huggingface_hub import login
    login(token=token, add_to_git_credential=False)
    print("[Cell 2] Authenticated with HuggingFace")
else:
    from huggingface_hub import login
    login()
    print("[Cell 2] Authenticated via interactive login")
