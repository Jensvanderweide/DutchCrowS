import pandas as pd
from tqdm import tqdm
from code.utils import compare_pair

def evaluate_likelihood(lm, data, sample_size=None, model_name=None):
    total = len(data.index)
    eval_data = data.sample(sample_size, random_state=42) if sample_size else data

    results = []
    total_stereo, total_antistereo = 0, 0
    stereo_score, antistereo_score = 0, 0
    N = 0
    neutral = 0
    print("Running likelihood evaluation...")

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
            results.append({
                "sent_more": entry["sent1"] if direction == "stereo" else entry["sent2"],
                "sent_less": entry["sent2"] if direction == "stereo" else entry["sent1"],
                "preferred": score["preferred"],
                "score": pair_score,
                "stereo_antistereo": direction,
                "bias_type": bias,
                "overlap": score["overlap"]
            })

    df_score = pd.DataFrame(results)
    df_score.to_csv(
        f"experiment_results/likelihood_evaluation_results_{model_name}_n={sample_size}.csv",
        index=False,
    )
    return df_score
