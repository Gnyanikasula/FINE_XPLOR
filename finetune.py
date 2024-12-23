import torch
import json
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from huggingface_hub import login
from unsloth import FastLanguageModel

token = "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"  # Replace with your Hugging Face token
login(token)




# Model and Quantization Configuration
model_name = "mistralai/Mistral-7B-v0.3"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=["lm_head"],
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=512,
    quantization_config=bnb_config,
    device_map="auto",
      # Enables fast downloading
)


# Apply LoRA Configuration
model = FastLanguageModel.get_peft_model(
    model,
    r=8,                        # Reduced rank for small datasets
    target_modules=["q_proj", "v_proj"],  # Focus on key layers for adaptation
    lora_alpha=64,              # Stronger updates
    lora_dropout=0.2,           # Reduce overfitting
)

# Load Dataset and Split into Train/Validation
dataset = load_dataset("gnyani007/data5000", split="train")
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split["train"]
val_dataset = train_test_split["test"]

# Tokenization Function
def tokenize_function(examples):
    inputs = [f"Context: Provide a precise answer for the financial related question, Question: {q} .\nAnswer:" for q in examples["question"]]
    targets = [a for a in examples["answer"]]
    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
    labels = tokenizer(targets, max_length=512, padding="max_length", truncation=True)["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs

# Tokenize Datasets
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["question", "answer"])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=["question", "answer"])






# Fine-Tuning Arguments
training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/FIN",
    per_device_train_batch_size = 16,
    gradient_accumulation_steps = 4,
    per_device_eval_batch_size = 16,
                                    # Effective batch size = 4 * 16 = 64
    learning_rate=3e-5,
    num_train_epochs=5,
    evaluation_strategy="epoch",  # Evaluate after each epoch
    save_strategy="epoch",        # Save model checkpoints per epoch
    save_total_limit=2,           # Keep only the 2 most recent checkpoints
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=50,
    load_best_model_at_end=True,
    report_to="none",
    fp16=True,                    # Mixed precision for faster training
    optim="paged_adamw_32bit",    # Optimizer for large models
)

# Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# Fine-Tune the Model
trainer.train()
# Save the fine-tuned model and tokenizer
output2_dir = ""
model.save_pretrained(output2_dir)
tokenizer.save_pretrained(output2_dir)

repo_id = "gnyani007/test23"

# Push model and tokenizer
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)


print(f"Model successfully pushed to Hugging Face Hub at: https://huggingface.co/{repo_id}")


