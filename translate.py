import pandas as pd
from deep_translator import GoogleTranslator
import csv
from tqdm import tqdm

def translate_to_dutch(text: str) -> str:
    """Translate the given text to Dutch using Google Translate."""
    translator = GoogleTranslator(source='en', target='nl')
    translation = translator.translate(text)
    return translation

def translate_data_to_dutch(data_path: str, output_path: str):
    """Translate the entire data to Dutch and save it to a new file."""
    data = pd.read_csv(data_path, delimiter="\t")

    # Initialize progress bar
    with tqdm(total=len(data) * 2, desc="Translating", unit="sentence") as pbar:
        # Translate each column in the data
        data['sent_more'] = data['sent_more'].apply(lambda x: translate_to_dutch(x) or pbar.update(1))
        print(data['sent_more'])
        return
        data['sent_less'] = data['sent_less'].apply(lambda x: translate_to_dutch(x) or pbar.update(1))

    # Save the translated data to a new file
    data.to_csv(output_path, index=False)

# Example usage
data_path = "crows_pairs_neveol_revised.csv"
output_path = "nl_crows_pairs_neveol_revised.csv"
translate_data_to_dutch(data_path, output_path)
print(f"Data has been translated and saved to {output_path}.")
