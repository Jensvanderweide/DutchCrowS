import pandas as pd
import csv

def read_data(input_file): 
    """Restructures the data by dividing the sentences into 'less' and 'more' bias.

    :param csv input_file: CrowS-Pairs-like csv-file
    :return DataFrame: restructured dataframe
    """
    df = pd.DataFrame(columns=['sent1', 'sent2', 'direction', 'bias_type'])
    
    with open(input_file, encoding="utf-8") as f: 
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader: 
            direction = row['stereo_antistereo']
            bias_type = row['bias_type']
            
            sent1, sent2 = '', ''
            if direction == 'stereo': 
                sent1 = row['sent_more']
                sent2 = row['sent_less']
            else: 
                sent1 = row['sent_less']
                sent2 = row['sent_more'] 
                
            df_item = {'sent1': sent1,
                       'sent2': sent2,
                       'direction': direction,
                       'bias_type': bias_type}
            df = df._append(df_item, ignore_index=True)

    preprocessed_file = "preprocessed_" + input_file.replace(".csv", "") + ".csv"
    df.to_csv(preprocessed_file, index=False, sep='\t')
    print(f"Saved restructured data to: {preprocessed_file}")
            
    return df 

def preprocess(input_file="final.csv"):
    """Wrapper for the preprocessing function."""
    return read_data(input_file)

if __name__ == "__main__":
    preprocess("final.csv")  # Run this for preprocessing the 'final.csv'
