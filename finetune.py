from datasets import load_dataset
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

# Hugging Face Login (with your Hugging Face token)
token = "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"  # Replace with your Hugging Face token
login(token)

# Load dataset from Hugging Face Hub
dataset = load_dataset("gnyani007/TRY_FIN", split="train")

# Tokenizer initialization
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.3", trust_remote_code=True)

# Ensure pad_token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

# Tokenize dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(['text'])

# DataLoader setup (optional, if DataLoader is needed later)
from torch.utils.data import DataLoader
train_loader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True)

# Model setup with quantization
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Set to True for 8-bit quantization
    llm_int8_threshold=6.0,  # Threshold for outlier detection
    llm_int8_skip_modules=["lm_head"],  # Skip quantizing output layers if needed
)

# Load the model with quantization config
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.3",
    quantization_config=bnb_config,
    device_map="auto"  # Automatically map model layers to available GPUs
)

# LoRA configuration and application
lora_config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.1, 
    bias="none", 
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="/kaggle/working/mistral-lora-output",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=3,
    learning_rate=2e-5,
    fp16=True,
    optim="adamw_8bit"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()
