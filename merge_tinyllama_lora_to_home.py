
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# Define paths
base_model_path = "/tmp/llm_models/TinyLlama-1.1B-Chat-v1.0"
lora_model_path = os.path.expanduser("~/models/tinyllama_finetuned")
output_path = os.path.expanduser("~/models/tinyllama_merged")

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype="auto",
    device_map="cpu"
)

# Load and merge LoRA adapter
model = PeftModel.from_pretrained(base_model, lora_model_path)
model = model.merge_and_unload()

# Save merged model
model.save_pretrained(output_path)

# Save tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(output_path)

print(f"✅ Merged TinyLlama model saved to {output_path}")
