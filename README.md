#  Bias Evaluation for Language Models

This project evaluates stereotyping bias in language models using two evaluation strategies:  
- **Likelihood-based evaluation**
- **Prompt-based evaluation**

It supports any HuggingFace-compatible model (e.g., `gpt2`, `xlm-roberta-base`, etc.) and takes preprocessed input data in `.csv` format.

---

## Project Structure

```
.
├── code/
│   ├── models.py            
│   ├── eval_likelihood.py    
│   ├── eval_prompt.py        
│   ├── utils.py              
├── data/
│   └── preprocessed_final.csv  
├── main.py                   
```

---

## Getting Started

### 1. Install Requirements

Make sure you have Python ≥ 3.8 and install dependencies:

```bash
pip install torch transformers pandas
```

### 2. Prepare Your Data

Ensure your dataset is a `.csv` file with tab separation (`\t`) and is located in the `data/` folder.

Example file path: `data/preprocessed_final.csv`

### 3. Run the Script

```bash
python main.py --model_name gpt2 --eval_mode likelihood
```

You can also specify:

- `--data_file`: Path to the dataset inside the `data/` folder (default: `preprocessed_final.csv`)
- `--sample_size`: Limit the number of rows to evaluate (optional)
- `--eval_mode`: Choose between `likelihood` or `prompt`

#### Example:

```bash
python main.py --model_name xlm-roberta-base --eval_mode prompt --sample_size 500
```

---

## Evaluation Modes

- **Likelihood mode**: Scores sentence pairs based on log-likelihood using a causal language model.
- **Prompt mode**: Uses a prompt template to assess model biases.

Each evaluation function will log or save relevant metrics such as:

- Stereotype scores
- Bias directionality

---

