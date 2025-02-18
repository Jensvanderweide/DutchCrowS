
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "openai-community/gpt2-medium"

# Load tokenizer and model without quantization
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare input
input_text = "What is the capital of France?"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate output
output = model.generate(**inputs)

# Decode and print result
print(tokenizer.decode(output[0], skip_special_tokens=True))
