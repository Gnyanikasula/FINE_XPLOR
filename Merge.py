import peft
import torch
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel
import os
from huggingface_hub import login
token = "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"  # Replace with your Hugging Face token
login(token)

# Paths for the LoRA weights and output directory
lora_path = "outputs_mistral/checkpoint-18200"  # Path to the LoRA checkpoint
output_path = "outputs_mistral/merged_model"   # Output path for merged model
model_name = "mistralai/Mistral-7B-v0.3"       # The base model name from Hugging Face

# Step 1: Load LoRA weights and configuration
peft_model_id = lora_path
peft_config = PeftConfig.from_pretrained(peft_model_id)

# Step 2: Load the base Mistral 7B model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    return_dict=True,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto",  # Automatically distribute model across available devices
    cache_dir="./models"
)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models")

# Step 3: Copy the base model files to the output directory
os.makedirs(output_path, exist_ok=True)
shutil.copytree(model_name, output_path, dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.pt', "*.pth", "*.bin"))

# Step 4: Load the LoRA model into the base model
model = PeftModel.from_pretrained(model, peft_model_id)
model.eval()  # Set the model to evaluation mode

# Step 5: Merge LoRA weights with the base model
key_list = [key for key, _ in model.named_modules() if "lora" not in key]
for key in key_list:
    try:
        sub_mod = model.get_submodule(key)
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
    except AttributeError:
        continue
    target_name = key.split(".")[-1]
    
    # Check if the submodule is an instance of a LoRA Linear layer
    if isinstance(sub_mod, peft.tuners.lora.Linear):
        sub_mod.merge()  # Merge the LoRA weights
        bias = sub_mod.bias is not None
        new_module = torch.nn.Linear(sub_mod.in_features, sub_mod.out_features, bias=bias)
        new_module.weight.data = sub_mod.weight  # Copy merged weights
        if bias:
            new_module.bias.data = sub_mod.bias  # Copy bias if present
        model.base_model._replace_module(parent, target_name, new_module, sub_mod)

# Step 6: Extract the base model after LoRA merge
model = model.base_model.model

# Step 7: Save the merged model and tokenizer to the output path
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"Merged model saved to {output_path}")
