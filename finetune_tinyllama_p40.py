
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, Dataset

# Paths
model_path = os.path.expanduser("./TinyLlama-1.1B-Chat-v1.0")
input_file = os.path.expanduser("./input.txt")
output_dir = os.path.expanduser("./tinyllama_finetuned")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

# LoRA Configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, peft_config)

# Load and tokenize data
def load_training_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return Dataset.from_dict({"text": lines})

dataset = load_training_data(input_file)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training Arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=10,                  # Log every 10 steps
    save_strategy="epoch",
    save_total_limit=1,
    fp16=True,
    report_to="none",                  # Disable wandb
    logging_dir=os.path.join(output_dir, "logs"),
    disable_tqdm=False
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training
print("🚀 Starting fine-tuning...")
trainer.train()
print("✅ Fine-tuning complete. Model saved to:", output_dir)
