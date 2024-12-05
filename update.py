from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import login
token = "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"  # Replace with your Hugging Face token
login(token)
# Configuration
device = "auto"  # Automatically map model to available devices (GPU/CPU)
local_model_path = "outputs_mistral/merged_model"  # Path to the merged weights (after LoRA merge)
repo_name = "gnyani007/TRY_model1"  # Your Hugging Face repo name (replace 'your_hf_username')
hf_token = "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"  # Your Hugging Face authentication token

# Load the merged model and tokenizer from the local path
model = AutoModelForCausalLM.from_pretrained(
    local_model_path, 
    trust_remote_code=True,  # This ensures remote code execution if needed (for non-standard models)
    device_map=device,  # Automatically map model to available devices
    torch_dtype=torch.float16,  # Load the model with FP16 precision (saves memory)
).eval()  # Set model to evaluation mode (disable dropout, etc.)
tokenizer = AutoTokenizer.from_pretrained(local_model_path)

# Push the model and tokenizer to the Hugging Face Hub
model.push_to_hub(repo_name, token=hf_token)
tokenizer.push_to_hub(repo_name, token=hf_token)

print(f"Model and tokenizer pushed to Hugging Face repo: {repo_name}")
