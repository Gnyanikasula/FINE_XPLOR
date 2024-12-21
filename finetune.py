import torch
import json
import numpy as np
from datasets import load_dataset
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader
from transformers import BitsAndBytesConfig, TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from huggingface_hub import login
from unsloth import FastLanguageModel

token = "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"  # Replace with your Hugging Face token
login(token)

model_name = "mistralai/Mistral-7B-v0.3"


# Model setup with quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Set to True for 8-bit quantization
    llm_int8_threshold=6.0,  # Threshold for outlier detection
    llm_int8_skip_modules=["lm_head"],  # Skip quantizing output layers if needed
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=512,        # Adjust sequence length based on your data
              # Enable 4-bit quantization
    quantization_config=bnb_config,
    device_map="auto",          # Automatically map model layers to available GPUs
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,                       # Rank of LoRA adapters
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target specific layers
    lora_alpha=32,
    lora_dropout=0.1
)

dataset = load_dataset('gnyani007/data5000', split='train')

# Tokenization
def tokenize_function(examples):
    # Prepare inputs and targets
    inputs = [f"Q: {q} A:" for q in examples['question']]
    targets = [f"{a}" for a in examples['answer']]
    
    # Debug: Print the first few inputs and targets
    # print("Inputs:", inputs[:3])  # Adjust the number to inspect more samples
    # print("Targets:", targets[:3])
    
    # Tokenize inputs and targets
    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
    labels = tokenizer(targets, max_length=512, padding="max_length", truncation=True)["input_ids"]
    
    # Debug: Print tokenized inputs and labels for the first few samples
    # print("Tokenized Inputs:", model_inputs['input_ids'][:3])
    # print("Tokenized Labels:", labels[:3])
    
    model_inputs["labels"] = labels
    return model_inputs

# Apply tokenization
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["question", "answer"]
)

# Debug: Print structure of the tokenized dataset
# print("Tokenized Dataset Example:", tokenized_dataset[0])



# Training arguments
output_dir = "/content/drive/MyDrive/FIN"
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=1,
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    gradient_checkpointing=True,
    fp16=True,
    optim="paged_adamw_32bit",
    report_to="none",
    eval_strategy="no"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
output2_dir = ""
model.save_pretrained(output2_dir)
tokenizer.save_pretrained(output2_dir)

repo_id = "gnyani007/testsquad"

# Push model and tokenizer
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)


print(f"Model successfully pushed to Hugging Face Hub at: https://huggingface.co/{repo_id}")


