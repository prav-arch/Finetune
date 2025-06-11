
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

# 1. Load dataset from input.txt
def load_dataset_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    return Dataset.from_list(data)

dataset = load_dataset_from_txt("/mnt/data/input.txt")  # update path if needed

# 2. Load base model & tokenizer
model_path = "/tmp/llm_models"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 3. Apply PEFT (LoRA)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

model = get_peft_model(model, peft_config)

# 4. Tokenize the dataset
def tokenize(example):
    full_prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    return tokenizer(full_prompt, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize)

# 5. Define training arguments
training_args = TrainingArguments(
    output_dir="/tmp/llm_models/fine-tuning",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=1,
    report_to="none"
)

# 6. Create Trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()

# 7. Save the fine-tuned model and tokenizer
model.save_pretrained("/tmp/llm_models/fine-tuning")
tokenizer.save_pretrained("/tmp/llm_models/fine-tuning")

print("✅ Fine-tuning complete. Model saved to /tmp/llm_models/fine-tuning")
