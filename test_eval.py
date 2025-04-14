import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval import read_data, overlap, compare_pair, evaluate
from openai.openai_eval_prompt import *
import math
from openai import OpenAI
import os

if __name__ == "__main__":

    # ======== LOAD TOKENIZER & MODEL ========
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    # ======== MOCK LANGUAGE MODEL OBJECT ========
    lm = {
        "model": model,
        "tokenizer": tokenizer,
        "softmax": torch.nn.Softmax(dim=-1),
        "log_softmax": torch.nn.LogSoftmax(dim=-1),
        "mask_token": tokenizer.bos_token,
        "uncased": False
    }

    # ======== DATA ========
    data = read_data("crows_pairs_neveol_revised.csv")

    # ======== TEST evaluate() ========
    evaluate(lm, data, sample_size=100, model_name="gpt2")
