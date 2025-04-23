import pandas as pd
from deep_translator import GoogleTranslator
from functools import lru_cache
from tqdm import tqdm




def translate_to_dutch(text: str) -> str:
    """Translate the given text to Dutch using Google Translate."""
    translator = GoogleTranslator(source='en', target='nl')
    translation = translator.translate(text)
    return translation

def translate_and_track(text, pbar):
    translation = translate_to_dutch(text)
    pbar.update(1)
    return translation 

@lru_cache(maxsize=None)
def translate_to_dutch_cached(text):
    return translate_to_dutch

def translate_data_to_dutch(data_path: str, output_path: str, n=None):
    """Translate the entire data to Dutch and save it to a new file."""
    data = pd.read_csv(data_path, delimiter="\t")
    if n:
        data = data.head(n)

    # Initialize progress bar
    with tqdm(total=len(data) * 2, desc="Translating", unit="sentence") as pbar:
        # Translate each column in the data
        data['sent_more'] = data['sent_more'].apply(lambda x: translate_and_track(x, pbar))
        data['sent_less'] = data['sent_less'].apply(lambda x: translate_and_track(x, pbar))

    # Save the translated data to a new file
    data.to_csv(output_path, sep='\t', index=False)

n = 1000
data_path = "crows_pairs_neveol_revised.csv"
output_path = f"translated_data/dutch_crows_pairs_neveol_revised_{n}.csv"


translate_data_to_dutch(data_path, output_path, n=n)
print(f"Data has been translated and saved to {output_path}.")
