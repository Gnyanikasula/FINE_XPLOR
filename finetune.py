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
    load_in_4bit=True,          # Enable 4-bit quantization
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

dataset = load_dataset("gnyani007/FIN_JSON", split="train")

def flatten_dataset(data):
    flattened_data = []
    for company_data in data:
        for pair in company_data["Query-Response Pairs"]:
            flattened_data.append({
                "question": pair["question"],
                "response": pair["response"],
                "quarterly_results": company_data.get("Quarterly Results", {}),
            })
    return flattened_data


from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3", trust_remote_code=True)

# Ensure pad_token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    # Tokenize the question and response independently
    inputs = tokenizer(
        examples['question'],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    targets = tokenizer(
        examples['response'],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    # Add labels to the inputs
    inputs["labels"] = targets["input_ids"]
    return inputs


def process_numerical_features_batch(quarterly_results_batch, metrics_list):
    # print(quarterly_results_batch)

    processed_features = []
    for quarterly_results in quarterly_results_batch:
        if not quarterly_results:
            # Handle missing quarterly results
            processed_features.append([0] * len(metrics_list))
            continue

        # Ensure quarterly_results is a dictionary
        if isinstance(quarterly_results, list):
            quarterly_results = {f"Q{index+1}": result for index, result in enumerate(quarterly_results)}

        if not isinstance(quarterly_results, dict):
            raise ValueError(f"Expected dictionary for quarterly_results but got {type(quarterly_results)}")

        features = []
        for quarter, metrics in quarterly_results.items():
            if not metrics:  # Check if 'metrics' is None
                metrics = {}
            for metric in metrics_list:
                try:
                    value = float(metrics.get(metric, 0)) if metrics.get(metric) is not None else 0
                    features.append(value)
                except (ValueError, TypeError):
                    features.append(0)
        processed_features.append(features)
    return processed_features


from datasets import Dataset as HFDataset

# Flatten the dataset
flattened_data = flatten_dataset(dataset)

# Convert to Hugging Face Dataset
hf_dataset = HFDataset.from_list(flattened_data)

# Metrics list for processing numerical features
metrics_list = [
    "Sales", "YOY Sales Growth %", "Expenses", "Material Cost %",
    "Employee Cost %", "Operating Profit", "OPM %", "Other Income",
    "Other income normal", "Interest", "Depreciation",
    "Profit before tax", "Tax %", "Net Profit", "EPS in Rs"
]

# Add tokenized data and numerical features
# Add tokenized data and numerical features
hf_dataset = hf_dataset.map(
    lambda batch: {
        **tokenize_function(batch),  # Handles batched tokenization
        "numerical_features": process_numerical_features_batch(batch['quarterly_results'], metrics_list),
    },
    batched=True,
    remove_columns=["question", "response", "quarterly_results"]
)



output_dir = "/content/drive/MyDrive/FIN"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,         # Reduced batch size for memory efficiency
    gradient_accumulation_steps=4,         # Simulate larger batch sizes
    learning_rate=2e-4,
    num_train_epochs=3,
    save_strategy= "epoch",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    gradient_checkpointing=True,           # Saves memory during training
    fp16=True,                             # Mixed precision for faster training
    optim="paged_adamw_32bit",             # Optimizer for memory efficiency
    report_to="none",                      # Disable external reporting
    eval_strategy="no",             # No evaluation during training
)

# Trainer setup
trainer = Trainer(
    model=model,                       # Your Mistral model
    args=training_args,                # Training arguments
    train_dataset=hf_dataset,          # Processed dataset
    tokenizer=tokenizer                # Tokenizer for preprocessing
)

trainer.train()


