from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
token = "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"  # Replace with your Hugging Face token
login(token)

# Model path and configuration
model_path = "outputs_squad"  # Path to your fine-tuned model directory
device = "auto"  # Use 'cuda' for GPU or 'cpu' for CPU

# LoRA quantization and configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map=device,
    quantization_config=bnb_config
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

# Initialize the text generation pipeline
pipeline_nlp = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1  # Set device for GPU (0) or CPU (-1)
)

def ask_finance_question(question: str) -> str:
    """
    Use the model pipeline to get answers for finance-related queries.
    :param question: The finance question to ask.
    :return: The model's answer.
    """
    prompt = f"Question: {question}\nAnswer:"
    response = pipeline_nlp(prompt, max_length=200, num_return_sequences=1, temperature=0.7)
    return response[0]['generated_text']

# Example query
if __name__ == "__main__":
    question = "What was the Sales in Sep 2024 for InfoBeans Technologies Limited (Information Technology, Computers - Software & Consulting)?"
    answer = ask_finance_question(question)
    print("Answer:", answer)
