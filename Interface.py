from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

from huggingface_hub import login
token = "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"  # Replace with your Hugging Face token
login(token)
import gradio as gr

# Path to the fine-tuned model
model_path = "outputs_squad/merged_model"

# Load model function with quantization configuration
def load_model(model_name):
    # Load model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        load_in_8bit=True,
        quantization_config=bnb_config,
        cache_dir="models"
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # Inference class
    class Infer:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer

        def forward(self, text, limit=128, temp=1.0):
            # Prepare input text
            text = self.tokenizer(text, return_tensors="pt").to("cuda")
            # Generate text using the model
            output = self.model.generate(
                **text,
                do_sample=True,
                temperature=temp,
                max_new_tokens=int(limit),
                top_p=0.95,
                top_k=60,
                pad_token_id=self.tokenizer.pad_token_id
            )
            # Decode output
            return self.tokenizer.decode(output[0], skip_special_tokens=True)

    return Infer(model, tokenizer)

# Load model
model = load_model(model_path)

# Define prediction function
def predict(temp, limit, text):
    prompt = text
    # Use the inference class for prediction
    out = model.forward(prompt, limit, temp)
    return out

# Create Gradio interface
pred = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(0.001, 10, value=0.1, label="Temperature"),
        gr.Slider(1, 1024, value=128, label="Token Limit"),
        gr.Textbox(
            label="Input",
            lines=1,
            value="#### Human: What's the capital of Australia?#### Assistant: ",
        ),
    ],
    outputs='text',
)

# Launch Gradio app
if __name__ == "__main__":
    pred.launch(share=True)
