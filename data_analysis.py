import pandas as pd
import matplotlib.pyplot as plt

def analyze_results(df_score, lang):
    # Calculate overall metrics
    total = len(df_score)
    stereo_score = df_score[df_score['stereo_antistereo'] == 'stereo']['score'].sum()
    antistereo_score = df_score[df_score['stereo_antistereo'] == 'antistereo']['score'].sum()
    neutral = df_score[df_score['preferred'] == 'neutral'].shape[0]
    total_stereo = df_score[df_score['stereo_antistereo'] == 'stereo'].shape[0]
    total_antistereo = df_score[df_score['stereo_antistereo'] == 'antistereo'].shape[0]

    print('=' * 100)
    print('STEREOTYPE EVALUATION RESULTS')
    print('=' * 100)
    print('Total examples:', total)
    print('Metric score:', round((stereo_score + antistereo_score) / total * 100, 2))
    print('Stereotype score:', round(stereo_score / total_stereo * 100, 2) if total_stereo > 0 else "N/A")
    print('Anti-stereotype score:', round(antistereo_score / total_antistereo * 100, 2) if total_antistereo > 0 else "N/A")
    print("Num. neutral:", neutral, round(neutral / total * 100, 2))
    print('total stereo: ', total_stereo)
    print('total anti-stereo: ', total_antistereo)
    print('=' * 100)

    # Distribution analysis
    group_distribution = df_score.groupby('bias_type').agg({
        'score': 'sum',
        'sent_more': 'count'
    }).rename(columns={'sent_more': 'total'}).reset_index()

    group_distribution['percentage'] = group_distribution['score'] / group_distribution['total'] * 100

    print('Distribution of scores per social group:')
    print(group_distribution)

    # Plotting the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(group_distribution['bias_type'], group_distribution['percentage'])
    plt.xlabel('Social Group')
    plt.ylabel('Metric Score')
    plt.title(f'Distribution of Scores per Social Group. Language: {lang}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig(f'analysis_results/{lang}_distribution_score_socialgroup.png')
    plt.show()

    return group_distribution

def analyze_impact_of_dataset_size(df_score, lang):
    increments = list(range(10, len(df_score) + 1, 10))
    metric_scores = []

    for size in increments:
        subset = df_score.iloc[:size]
        stereo_score = subset[subset['stereo_antistereo'] == 'stereo']['score'].sum()
        antistereo_score = subset[subset['stereo_antistereo'] == 'antistereo']['score'].sum()
        total = len(subset)
        metric_score = (stereo_score + antistereo_score) / total * 100
        metric_scores.append(metric_score)

    plt.figure(figsize=(10, 6))
    plt.plot(increments, metric_scores, marker='o')
    plt.xlabel('Number of Data Points')
    plt.ylabel('Metric Score')
    plt.title(f'Impact of Dataset Size on Metric Score. Language: {lang}')
    plt.grid(True)
    plt.savefig(f'analysis_results/impact_of_size_{lang}.png')
    plt.show()

if __name__ == "__main__":
    
    lang = 'nl'

    if lang == 'nl':
        path = "prompt_result\dutch_ollama_evaluation_results_prompt_mistral_7b_1000_temp0.2.csv"
    else: 
        path = "prompt_result\ollama_evaluation_results_prompt_mistral_7b_1000_temp0.2.csv"

    df_score = pd.read_csv(path, sep='\t')

    analyze_results(df_score, lang=lang)
    analyze_impact_of_dataset_size(df_score, lang=lang)

    
