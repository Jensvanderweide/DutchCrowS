import os
import pandas as pd
import matplotlib.pyplot as plt

def analyze_results(df, lang):
    total = len(df)

    # Sum of scores by type
    stereo = df.loc[df.stereo_antistereo == 'stereo', 'score'].sum()
    antistereo = df.loc[df.stereo_antistereo == 'antistereo', 'score'].sum()
    neutral_count = (df.preferred == 'neutral').sum()

    # Counts by type
    n_stereo = (df.stereo_antistereo == 'stereo').sum()
    n_antistereo = (df.stereo_antistereo == 'antistereo').sum()

    overall_metric = (stereo + antistereo) / total * 100 if total else 0
    stereo_pct     = stereo / n_stereo * 100 if n_stereo else float('nan')
    antistereo_pct = antistereo / n_antistereo * 100 if n_antistereo else float('nan')
    neutral_pct    = neutral_count / total * 100 if total else 0

    print('='*80)
    print('STEREOTYPE EVALUATION RESULTS')
    print('='*80)
    print(f'Total examples:          {total}')
    print(f'Overall metric score:    {overall_metric:.2f}%')
    print(f'Stereotype score:        {stereo_pct:.2f}%  ({n_stereo} examples)')
    print(f'Anti-stereotype score:   {antistereo_pct:.2f}%  ({n_antistereo} examples)')
    print(f'Neutral examples:        {neutral_count} ({neutral_pct:.2f}%)')
    print('='*80)

    # Distribution per bias_type
    dist = (
        df
        .groupby('bias_type')
        .agg(
            total_examples = ('score','count'),
            sum_scores     = ('score','sum')
        )
        .assign(
            pct_score = lambda d: 100 * d.sum_scores / d.total_examples
        )
        .reset_index()
    )

    print('\nDistribution of scores per social group:')
    print(dist)

    # Plot
    plt.figure(figsize=(10,6))
    plt.bar(dist.bias_type, dist.pct_score)
    plt.xlabel('Social Group (bias_type)')
    plt.ylabel('Score Percentage')
    plt.title(f'Distribution by Social Group — {lang}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs('analysis_results', exist_ok=True)
    plt.savefig(f'analysis_results/{lang}_distribution_socialgroup.png')
    plt.show()

    return dist

def analyze_impact_of_dataset_size(df, lang):
    sizes = list(range(10, len(df)+1, 10))
    metrics = []

    for n in sizes:
        subset = df.iloc[:n]
        s = subset.loc[subset.stereo_antistereo == 'stereo','score'].sum()
        a = subset.loc[subset.stereo_antistereo == 'antistereo','score'].sum()
        metric = (s + a) / n * 100
        metrics.append(metric)

    plt.figure(figsize=(10,6))
    plt.plot(sizes, metrics, marker='o')
    plt.xlabel('Number of Data Points')
    plt.ylabel('Overall Metric Score (%)')
    plt.title(f'Impact of Dataset Size on Metric — {lang}')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs('analysis_results', exist_ok=True)
    plt.savefig(f'analysis_results/impact_size_{lang}.png')
    plt.show()

if __name__ == '__main__':
    lang = 'nl'
    filename = f"dutch_ollama_evaluation_results_prompt_mistral_7b_1000_temp0.2.csv" if lang == 'nl' \
               else "ollama_evaluation_results_prompt_mistral_7b_1000_temp0.2.csv"
    path = os.path.join("prompt_result", filename)

    df = pd.read_csv(path, sep='\t')
    analyze_results(df, lang)
    analyze_impact_of_dataset_size(df, lang)
