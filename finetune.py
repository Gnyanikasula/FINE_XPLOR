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
import wandb
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

token = "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"  # Replace with your Hugging Face token
login(token)


max_seq_length = 2048
dtype = None
load_in_4bit = True



model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = "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",
)


model = FastLanguageModel.get_peft_model(
    model,
    r = 8,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0.2,
    bias = "lora_only",

    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = True,
    loftq_config = None,
)


from datasets import load_dataset
from datasets import DatasetDict

# Define the Alpaca prompt
alpaca_prompt = """You are an AI trained on financial data. Answer the following question accurately.

### Question:
{}

### Response:
{}

"""

EOS_TOKEN = tokenizer.eos_token  # End of sentence token

# Formatting function to structure the prompts
def formatting_prompts_func(examples):
    questions = examples["question"]
    answers = examples["answer"]
    texts = []
    for question, answer in zip(questions, answers):
        text = alpaca_prompt.format(question, answer) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

# Load dataset from Hugging Face Hub
dataset = load_dataset("gnyani007/data5000", split="train")

# Split the dataset into training and validation sets (e.g., 80-20 split)
train_test_split = dataset.train_test_split(test_size=0.2, seed=42)
split_dataset = DatasetDict({
    "train": train_test_split["train"],
    "validation": train_test_split["test"],
})

# Apply the formatting function to both splits
formatted_split_dataset = split_dataset.map(formatting_prompts_func, batched=True)

# # Save the formatted datasets for fine-tuning
# formatted_split_dataset["train"].save_to_disk("formatted_train_dataset")
# formatted_split_dataset["validation"].save_to_disk("formatted_validation_dataset")




# Initialize the trainer with tuned parameters
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_split_dataset["train"],
    eval_dataset=formatted_split_dataset["validation"],
    dataset_text_field="text",
    max_seq_length=256,
    dataset_num_proc=1,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        max_steps=300,
        learning_rate=3e-5,
        fp16=True,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        output_dir="/content/drive/MyDrive/FIN",
        logging_dir="/content/drive/MyDrive/FIN/LOG",
        report_to=["wandb"],
        evaluation_strategy="steps",
        eval_steps=10,
        save_strategy="steps",
        save_steps=10,
        load_best_model_at_end=True,
    ),
)


trainer_stats = trainer.train()

# Save the fine-tuned model and tokenizer
output2_dir = ""
model.save_pretrained(output2_dir)
tokenizer.save_pretrained(output2_dir)

repo_id = "gnyani007/test24"

# Push model and tokenizer
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)


print(f"Model successfully pushed to Hugging Face Hub at: https://huggingface.co/{repo_id}")
