
from huggingface_hub import snapshot_download
#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
# Download the Mistral 7B Instruct v0.2 model to /tmp/llm_models
snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    local_dir="/tmp/llm_models",
    local_dir_use_symlinks=False
)

print("✅ Model downloaded to /tmp/llm_models")
