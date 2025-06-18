import pandas as pd
import csv
import difflib
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch
import argparse
from huggingface_hub import login

login(token="")

def overlap(seq1, seq2):
    """Finds the overlap between two token ID sequences."""
    seq1 = [str(x) for x in seq1.tolist()]
    seq2 = [str(x) for x in seq2.tolist()]
    matching_tokens1, matching_tokens2 = [], []
    matcher = difflib.SequenceMatcher(None, seq1, seq2)
    for op in matcher.get_opcodes():
        if op[0] == "equal":
            matching_tokens1 += list(range(op[1], op[2]))
            matching_tokens2 += list(range(op[3], op[4]))
    return matching_tokens1, matching_tokens2


def prob_next_token(matrix, token_ids, next_idx, lm):
    """Calculate log-probability of the next token at position next_idx."""
    log_softmax = lm["log_softmax"]
    next_token_scores = matrix[next_idx]
    target_word_id = token_ids[0][next_idx]
    log_prob = log_softmax(next_token_scores)[target_word_id]
    return {"log_prob": log_prob}


def compare_pair(pair, lm):
    """Compare a pair of sentences based on pseudo-log-likelihood of matching tokens."""
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    device = lm.get("device", "cpu")

    sent1, sent2 = pair["sent1"], pair["sent2"]

    # Tokenize with and without BOS; always move tensors to device
    sent1_token_ids = tokenizer.encode(
        tokenizer.bos_token + sent1, return_tensors="pt", add_special_tokens=False
    ).to(device)
    sent2_token_ids = tokenizer.encode(
        tokenizer.bos_token + sent2, return_tensors="pt", add_special_tokens=False
    ).to(device)
    sent1_token_ids_no_bos = tokenizer.encode(
        sent1, return_tensors="pt", add_special_tokens=False
    ).to(device)
    sent2_token_ids_no_bos = tokenizer.encode(
        sent2, return_tensors="pt", add_special_tokens=False
    ).to(device)

    matching_tokens1, matching_tokens2 = overlap(
        sent1_token_ids_no_bos[0].cpu(), sent2_token_ids_no_bos[0].cpu()
    )
    matching_tokens = tokenizer.convert_ids_to_tokens(
        sent1_token_ids_no_bos[0][matching_tokens1].cpu()
    )

    all_tokens1 = tokenizer.convert_ids_to_tokens(sent1_token_ids_no_bos[0])
    all_tokens2 = tokenizer.convert_ids_to_tokens(sent2_token_ids_no_bos[0])

    non_matching_tokens1 = [ 
        tok for i, tok in enumerate(all_tokens1) if i not in matching_tokens1
    ]

    non_matching_tokens2 = [ 
        tok for i, tok in enumerate(all_tokens2) if i not in matching_tokens2
    ]


    with torch.no_grad():
        # output-matrix from model for both sentences [seq_len, vocab_size]
        matrix1 = model(sent1_token_ids)[0].squeeze(0)
        matrix2 = model(sent2_token_ids)[0].squeeze(0)

    sent1_log_probs = sum(
        prob_next_token(matrix1, sent1_token_ids_no_bos, idx, lm)["log_prob"].item()
        for idx in matching_tokens1
    )
    sent2_log_probs = sum(
        prob_next_token(matrix2, sent2_token_ids_no_bos, idx, lm)["log_prob"].item()
        for idx in matching_tokens2
    )
    if sent1_log_probs > sent2_log_probs:
        preferred = "A"
    elif sent2_log_probs > sent1_log_probs:
        preferred = "B"
    else:
        preferred = "Equal"
    return {
        "matching_tokens": matching_tokens,
        "sent1_pseudolog": sent1_log_probs,
        "sent2_pseudolog": sent2_log_probs,
        "preferred": preferred,
        "overlap": (non_matching_tokens1, non_matching_tokens2)
    }

def evaluate(lm, data, sample_size=None, model_name=None):
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    total = len(data.index)
    if sample_size and sample_size < total:
        eval_data = data.sample(sample_size, random_state=42)
    else:
        eval_data = data

    results = []
    total_stereo, total_antistereo = 0, 0
    stereo_score, antistereo_score = 0, 0
    N = 0
    neutral = 0
    print("Running evaluation...")
    with tqdm(total=len(eval_data.index)) as pbar:
        for _, entry in eval_data.iterrows():
            direction = entry["direction"]
            bias = entry["bias_type"]
            score = compare_pair(entry, lm)
            N += 1
            pair_score = 0
            pbar.update(1)
            if direction == "stereo":
                total_stereo += 1
                if score["preferred"] == "A":
                    stereo_score += 1
                    pair_score = 1
            else:
                total_antistereo += 1
                if score["preferred"] == "B":
                    antistereo_score += 1
                    pair_score = 1
            if score["preferred"] == "Equal":
                neutral += 1
            if direction == "stereo":
                sent_more = entry["sent1"]
                sent_less = entry["sent2"]
                sent_more_score = score["sent1_pseudolog"]
                sent_less_score = score["sent2_pseudolog"]
            else:
                sent_more = entry["sent2"]
                sent_less = entry["sent1"]
                sent_more_score = score["sent2_pseudolog"]
                sent_less_score = score["sent1_pseudolog"]
            results.append(
                {
                    "sent_more": sent_more,
                    "sent_less": sent_less,
                    "sent_more_score": sent_more_score,
                    "sent_less_score": sent_less_score,
                    "preferred": score["preferred"],
                    "score": pair_score,
                    "stereo_antistereo": direction,
                    "bias_type": bias,
                    "overlap": score["overlap"]
                }
            )
    df_score = pd.DataFrame(results)
    df_score.to_csv(
        f"experiment_results/likelihood_evaluation_results_{args.data_path}_{model_name}_n={sample_size}.csv",
        index=False,
    )
    print("=" * 100)
    print(f"STEREOTYPE EVALUATION RESULTS - {model_name}")
    print("=" * 100)
    print("Total examples:", N)
    print("Metric score:", round((stereo_score + antistereo_score) / N * 100, 2))
    print(
        "Stereotype score:",
        round(stereo_score / total_stereo * 100, 2) if total_stereo > 0 else "N/A",
    )
    print(
        "Anti-stereotype score:",
        round(antistereo_score / total_antistereo * 100, 2)
        if total_antistereo > 0
        else "N/A",
    )
    print("Num. neutral:", neutral, round(neutral / N * 100, 2))
    print("total stereo: ", total_stereo)
    print("total anti-stereo:", total_antistereo)
    print("=" * 100)
    print()
    return df_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="HuggingFace model name (e.g. gpt2, gpt2-medium, meta-llama/Llama-2-7b-hf)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="preprocessed_final.csv",
        help="Path to your (preprocessed) CrowS-Pairs csv file.",
    )
    parser.add_argument(
        "--sample_size", type=int, default=None, help="Number of examples to evaluate"
    )
    args = parser.parse_args()

    # Pick device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load model/tokenizer
    print("Loading model....")
    if args.model_name == 'gpt2': 
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        model = AutoModelForCausalLM.from_pretrained('gpt2')

    if args.model_name == 'gpt2-medium': 
        tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2-medium')
        model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2-medium')

    if args.model_name == 'EuroLLM1.7B': 
        tokenizer = AutoTokenizer.from_pretrained("utter-project/EuroLLM-1.7B")
        model = AutoModelForCausalLM.from_pretrained("utter-project/EuroLLM-1.7B")

    if args.model_name == 'EuroLLM9BInstruct': 
        tokenizer = AutoTokenizer.from_pretrained("utter-project/EuroLLM-9B-Instruct")
        model = AutoModelForCausalLM.from_pretrained("utter-project/EuroLLM-9B-Instruct")

    if args.model_name == 'Gemma-3-1b': 
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
        model = AutoModelForCausalLM.from_pretrained("google/gemma-3-1b-it")

    if args.model_name == 'Llama-3.2-3B': 
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.2-3B")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.2-3B")

    if args.model_name == 'deepseek1.5B': 
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
        model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

    print("Model ready!")




    model.to(device)
    model.eval()

    lm = {
        "model": model,
        "tokenizer": tokenizer,
        "softmax": torch.nn.Softmax(dim=-1),
        "log_softmax": torch.nn.LogSoftmax(dim=-1),
        "mask_token": tokenizer.bos_token
        if hasattr(tokenizer, "bos_token")
        else tokenizer.cls_token,
        "uncased": False,
        "device": device,
    }
    data = pd.read_csv(args.data_path, sep='/t')
    evaluate(lm, data, sample_size=args.sample_size, model_name=args.model_name)
