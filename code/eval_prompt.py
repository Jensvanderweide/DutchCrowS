import random
import torch
from tqdm import tqdm
from code.utils import compare_pair_prompt_batch

def evaluate_prompt(lm, data, sample_size=None, model_name=None, N_REPEATS=5, BATCH_SIZE=8):
    total = len(data.index)
    eval_data = data.sample(sample_size, random_state=42) if sample_size else data

    results = []
    print("Running prompt-based evaluation...")

    with tqdm(total=len(eval_data.index)) as pbar:
        for _, entry in eval_data.iterrows():
            direction = entry["direction"]
            bias = entry["bias_type"]

            preferences = []
            raw_outputs = []
            stereo_scores = []

            for i in range(0, N_REPEATS, BATCH_SIZE):
                batch_entries = [entry] * min(BATCH_SIZE, N_REPEATS - i)
                scores = compare_pair_prompt_batch(batch_entries, lm)

                for score in scores:
                    preferences.append(score["preferred"])
                    raw_outputs.append(score["raw_output"])

                    if direction == "stereo":
                        stereo_scores.append(1 if score["preferred"] == "A" else 0 if score["preferred"] == "B" else "Equal")
                    else:
                        stereo_scores.append(1 if score["preferred"] == "B" else 0 if score["preferred"] == "A" else "Equal")

            pref_counts = {label: preferences.count(label) for label in ["A", "B", "Equal"]}
            majority_preference = max(set(preferences), key=preferences.count)

            results.append({
                "sent_more": entry["sent1"] if direction == "stereo" else entry["sent2"],
                "sent_less": entry["sent2"] if direction == "stereo" else entry["sent1"],
                "bias_type": bias,
                "stereo_antistereo": direction,
                "preferences": preferences,
                "preference_counts": pref_counts,
                "majority_preference": majority_preference,
                "stereo_score_list": stereo_scores,
                "raw_outputs": raw_outputs
            })

            pbar.update(1)

    df_score = pd.DataFrame(results)
    df_score.to_json(
        f"experiment_results/prompt_evaluation_results_{model_name}_n={sample_size}_repeats={N_REPEATS}.jsonl",
        orient='records',
        lines=True
    )
    return df_score
