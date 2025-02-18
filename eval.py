import pandas as pd
import csv
import difflib
from transformers import AutoTokenizer

def read_data(input_file): 
    """resturctures the data by dividng the sentences into 'less' and 'more' bias

    :param csv input_file: CrowS-Pairs-like csv-file
    :return DataFrame: restructured dataframe
    """
    
    df = pd.DataFrame(columns=['sent1', 'sent2', 'direction', 'bias_type'])
    
    with open(input_file, encoding="utf-8") as f: 
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader: 
            direction, gold_bias = '_', '_'
            direction = row['stereo_antistereo']
            bias_type = row['bias_type']
            
            snet1, sent2= '', ''
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
            
    return df 

def overlap(seq1, seq2):
    """finds the overlap between two sequences 
    :param tensor seq1: 2D tensor with token IDs
    :param tensor seq2: 2D tensor with token IDs
    :return tuple:  containing two lists of token indices that match
    """
    
    seq1 = [str(x) for x in seq1.tolist()]
    seq2 = [str(x) for x in seq2.tolist()]
    
    matching_tokens1, matching_tokens2 = [], []
    matcher = difflib.SequenceMatcher(None, seq1, seq2)

    # find the matching token IDs and the corresponding token indices
    for op in matcher.get_opcodes():
        if op[0] == 'equal':
            matching_tokens1 += [x for x in range(op[1],op[2],1)]
            matching_tokens2 += [x for x in range(op[3],op[4],1)]

    return matching_tokens1, matching_tokens2

def prob_next_token(matrix, token_ids, next_idx, lm):

    model = lm["model"]
    tokenizer = lm["tokenizer"]
    softmax = lm["softmax"]
    log_softmax = lm["log_softmax"]
    uncased = lm["uncased"]
    
    # Get model scores only for the next token [NOT MINE]
    next_token_scores = matrix[next_idx]
    # Get score for word/token whose log-prob is being calculated {NOT MINE]
    target_word_id = token_ids[0][next_idx]
    # Use log_softmax layer to convert model scores for masked token to [NOT MINE]
    # log-prob. Then, log-prob for target word token id
    log_prob = log_softmax(next_token_scores)[target_word_id]

    return {'log_prob':log_prob}

def compare_pair(pair, lm): 
    
    model = lm["model"]
    tokenizer = lm["tokenizer"]
    
    sent1, sent2 = pair['sent_more'], pair['sent_less']
    
    # convert sentences to token IDs, one with BOS token for putting in the model and one without for finding overlap
    sent1_token_ids = tokenizer.encode(tokenizer.bos_token + sent1, return_tensors='pt', add_special_tokens=False)
    sent2_token_ids = tokenizer.encode(tokenizer.bos_token + sent2, return_tensors='pt', add_special_tokens=False)
    sent1_token_ids_no_bos = tokenizer.encode(sent1, return_tensors='pt', add_special_tokens=False)
    sent2_token_ids_no_bos = tokenizer.encode(sent2, return_tensors='pt', add_special_tokens=False)
    
   
    matching_tokens1, matching_tokens2 = overlap(sent1_token_ids_no_bos[0], sent2_token_ids_no_bos[0])
    match_no = len(matching_tokens1)
    matching_tokens = tokenizer.convert_ids_to_tokens(sent1_token_ids_no_bos[0][matching_tokens1])
    
    # output-matrix from model for both sentences
    matrix1 = model(sent1_token_ids)[0].squeeze(0)
    matrix2 = model(sent2_token_ids)[0].squeeze(0)
    
    # sentence scores by iterating over all overlapping tokens
    sent1_log_probs = sum(prob_next_token(matrix1, sent1_token_ids_no_bos, idx, lm)['log_prob'].item() for idx in matching_tokens1)
    sent2_log_probs = sum(prob_next_token(matrix2, sent2_token_ids_no_bos, idx, lm)['log_prob'].item() for idx in matching_tokens2)
    
    
    return {
        'matching_tokens': matching_tokens,
        'sent1_pseudolog': sent1_log_probs,
        'sent2_pseudolog': sent2_log_probs
    }

