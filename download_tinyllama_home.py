
from huggingface_hub import snapshot_download
import os

# Set local directory to ~/models/TinyLlama-1.1B-Chat-v1.0
output_dir = os.path.expanduser("~/models/TinyLlama-1.1B-Chat-v1.0")

# Download the model
snapshot_download(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    local_dir=output_dir,
    local_dir_use_symlinks=False
)

print(f"✅ Download complete: {output_dir}")
