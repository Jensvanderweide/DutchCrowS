from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name, device):
    """Load model and tokenizer from HuggingFace."""
    print("Loading model....")

    model_name_map = {
        "gpt2": "gpt2",
        "gpt2-medium": "openai-community/gpt2-medium",
        "EuroLLM1.7B": "utter-project/EuroLLM-1.7B",
        "EuroLLM9BInstruct": "utter-project/EuroLLM-9B-Instruct",
        "Gemma-3-1b": "google/gemma-3-1b-it",
        "Llama-3.2-3B": "meta-llama/Meta-Llama-3.2-3B",
        "deepseek1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "bloomz7b1-mt": "bigscience/bloomz-7b1-mt",
        "mistral7b-instruct-v0.1": "mistralai/Mistral-7B-Instruct-v0.1",
        "gemma-3-4b-it": "google/gemma-3-4b-it",
        "DeepSeek-R1-Distill-Qwen-7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    }

    if model_name not in model_name_map:
        raise ValueError(f"Unsupported model name: {model_name}")

    model_path = model_name_map[model_name]
    
    # Load tokenizer and model based on model_name_map
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Move model to the appropriate device (CPU or GPU)
    model.to(device)

    print(f"Loaded {model_name} model and tokenizer successfully!")

    return tokenizer, model

    
    if model_name not in model_name_map:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_map[model_name])
    model = AutoModelForCausalLM.from_pretrained(model_name_map[model_name], device_map="auto")
    
    return tokenizer, model

def prepare_lm(model, tokenizer, device):
    """Prepare the language model dictionary."""
    return {
        "model": model,
        "tokenizer": tokenizer,
        "softmax": torch.nn.Softmax(dim=-1),
        "log_softmax": torch.nn.LogSoftmax(dim=-1),
        "mask_token": tokenizer.bos_token if hasattr(tokenizer, "bos_token") else tokenizer.cls_token,
        "uncased": False,
        "device": device,
    }
