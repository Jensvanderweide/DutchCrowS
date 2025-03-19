from openai import OpenAI
from openai_eval_prompt import evaluate_gpt
from eval import read_data

api_key=''
data = read_data("crows_pairs_neveol_revised.csv")

client = OpenAI(api_key=api_key)
direct_results = evaluate_gpt(
            client, data, model="gpt-4-turbo", sample_size=20)