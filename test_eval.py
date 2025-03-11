import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval import read_data, overlap, compare_pair, evaluate
import math

# ======== LOAD TOKENIZER & MODEL ========
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

# # ======== SAMPLE SENTENCES ========
# sent_more = "That Sarah was cooked by a delicious dinner"
# sent_less = "That delicious dinner was cooked by Sarah"
# print("Sentence 1: ", sent_more)
# print("Sentence 2: ", sent_less)


# # ======== TEST overlap() ========
# sent1_token_ids = tokenizer.encode(sent_more, return_tensors='pt', add_special_tokens=False)[0]
# sent2_token_ids = tokenizer.encode(sent_less, return_tensors='pt', add_special_tokens=False)[0]

# matching_tokens1, matching_tokens2 = overlap(sent1_token_ids, sent2_token_ids)
# print("\nMatching token indices from overlap():")
# print("Matching Tokens Sent1:", matching_tokens1)
# print("Matching Tokens Sent2:", matching_tokens2)

# assert len(matching_tokens1) == len(matching_tokens2), "overlap() outputs should have the same length."
# assert len(matching_tokens1) > 0, "overlap() should detect shared tokens."

# # ======== TEST compare_pair() ========
# pair = {
#     'sent_more': sent_more,
#     'sent_less': sent_less
# }

# score = compare_pair(pair, lm)
# print("\nScore from compare_pair():")
# print("Matching Tokens:", score['matching_tokens'])
# print("Sent1 Pseudo-log-prob:", score['sent1_pseudolog'])
# print("Sent2 Pseudo-log-prob:", score['sent2_pseudolog'])
# print('')
# prob1 = math.exp(score['sent1_pseudolog'])
# prob2 = math.exp(score['sent2_pseudolog'])
# print(f"Probability of Sentence 1: {prob1}")
# print(f"Probability of Sentence 2: {prob2}")

# assert 'matching_tokens' in score, "compare_pair() output should include matching_tokens."
# assert 'sent1_pseudolog' in score, "compare_pair() output should include sent1_pseudolog."
# assert 'sent2_pseudolog' in score, "compare_pair() output should include sent2_pseudolog."

# ======== TEST evaluate() ========
data = read_data("crows_pairs_neveol_revised.csv")
evaluate(lm, data)
