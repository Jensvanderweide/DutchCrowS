from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
import torch

model_name = "EleutherAI/gpt-neo-125M"

# Create quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config
)

input_text = "What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="pt")

output = model.generate(**inputs)
print(tokenizer.decode(output[0], skip_special_tokens=True))