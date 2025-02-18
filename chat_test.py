import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "openai-community/gpt2-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def chat(): 
    print("GPT-2 medium chatbot (type 'exit' to quit)")
    
    while True: 
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break 
        
        inputs = tokenizer(user_input, return_tensors="pt")
        with torch.no_grad():
            output = model.generate(**inputs, max_length=100, 
                                    pad_token_id=tokenizer.eos_token_id,
                                    repetition_penalty = 1.2)
            
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        bot_response = response[len(user_input):].strip()
        
        print(f"Bot: {bot_response}\n")
        
if __name__ == "__main__":
    chat()