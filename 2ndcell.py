# Cell 2 - Authenticate with HuggingFace (Local PC)
# You need a HuggingFace token with access to google/medgemma-4b-it
# Get one at: https://huggingface.co/settings/tokens

import os

token = os.environ.get("HF_TOKEN")

# Try reading from a .env file in the current directory
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
    # Interactive login - will prompt you to paste your token
    from huggingface_hub import login
    login()
    print("[Cell 2] Authenticated via interactive login")