#!/usr/bin/env python
import argparse
import pickle
import numpy as np
import pandas as pd
import nltk
from evaluate import load
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from transformers import AutoTokenizer

# Download necessary nltk data
nltk.download('wordnet')
nltk.download('punkt')

def parse_args():
    parser = argparse.ArgumentParser(description="QA Evaluation Script with Traditional and Conditional BERTScore")
    parser.add_argument('--pickle_file', type=str, default="retrieved_data.pkl", help="Path to the pickle file containing evaluation data")
    parser.add_argument('--model_type', type=str, default="distilbert-base-multilingual-cased", help="Model type for BERTScore evaluation")
    # Optional CSV for conditional BERTScore evaluation:
    parser.add_argument('--csv_file', type=str, default=None, help="(Optional) Path to CSV file containing original data for conditional evaluation")
    parser.add_argument('--context_col', type=str, default="Chunk", help="Column name for original context in CSV")
    parser.add_argument('--question_col', type=str, default="question", help="Column name for question in CSV")
    parser.add_argument('--answer_col', type=str, default="answer", help="Column name for reference answer in CSV")
    parser.add_argument('--generated_col', type=str, default="jais_adapted_rag", help="Column name for generated answer in CSV (for conditional eval)")
    return parser.parse_args()

def load_pickle_data(pickle_file):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    actual_answers = data['actual_answers_of_queried_questions']
    generated_answers = data['generated_answers']
    retrieved_context_ids = data['retrieved_context_ids']
    return actual_answers, generated_answers, retrieved_context_ids

def compute_traditional_metrics(generated_answers, actual_answers, model_type):
    bertscore = load("bertscore")
    bert_results = bertscore.compute(predictions=generated_answers, references=actual_answers, model_type=model_type)
    precision = np.mean(bert_results['precision'])
    recall = np.mean(bert_results['recall'])
    f1 = np.mean(bert_results['f1'])
    print("=== Traditional BERTScore ===")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("Test Size:", len(bert_results['precision']))

    # BLEU and ROUGE-L evaluation
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    bleu_scores = []
    rougeL_scores = []
    for actual, generated in zip(actual_answers, generated_answers):
        bleu_score = sentence_bleu([actual.split()], generated.split())
        bleu_scores.append(bleu_score)
        rougeL_score = rouge_scorer_obj.score(actual, generated)['rougeL'].fmeasure
        rougeL_scores.append(rougeL_score)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)
    print("Average BLEU Score:", avg_bleu)
    print("Average ROUGE-L Score:", avg_rougeL)
    
    rouge = load("rouge")
    rouge_results = rouge.compute(predictions=generated_answers, references=actual_answers, tokenizer=lambda x: x.split())
    print("ROUGE Results:", rouge_results)

def compute_conditional_bertscore(csv_file, retrieved_context_ids, model_type):
    # Load CSV file containing original data
    df = pd.read_csv(csv_file)
    # We assume that the order of rows in the CSV aligns with the order of items in the pickle file.
    # Compute 'retrieved_context' using the retrieved_context_ids:
    df['retrieved_context'] = df.apply(lambda row: df.loc[row.name, df.columns[df.columns.str.contains('Chunk', case=False)].tolist()[0]]
                                       if retrieved_context_ids and row.name < len(retrieved_context_ids) else "", axis=1)
    
    # Initialize BERTScore and tokenizer
    bertscore = load("bertscore")
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    
    # Preprocess inputs: concatenate context, question, and answer for reference and retrieved context, question, and generated answer for candidate.
    def preprocess_inputs(row):
        ref = tokenizer.sep_token.join([str(row[df.columns[df.columns.str.contains('Chunk', case=False)].tolist()[0]]),
                                        str(row['question']),
                                        str(row['answer'])])
        cand = tokenizer.sep_token.join([str(row['retrieved_context']),
                                         str(row['question']),
                                         str(row['jais_adapted_rag'])])
        return pd.Series([ref, cand])
    
    df[['processed_references', 'processed_generated']] = df.apply(preprocess_inputs, axis=1)
    
    # Compute conditional BERTScore
    cond_results = bertscore.compute(predictions=df['processed_generated'].tolist(), 
                                     references=df['processed_references'].tolist(), 
                                     model_type=model_type)
    cond_precision = np.mean(cond_results['precision'])
    cond_recall = np.mean(cond_results['recall'])
    cond_f1 = np.mean(cond_results['f1'])
    print("=== Conditional BERTScore ===")
    print("Conditional Precision:", cond_precision)
    print("Conditional Recall:", cond_recall)
    print("Conditional F1:", cond_f1)
    print("Conditional Test Size:", len(cond_results['precision']))

def main():
    args = parse_args()
    
    # Load data from pickle file
    actual_answers, generated_answers, retrieved_context_ids = load_pickle_data(args.pickle_file)
    print("=== Traditional Evaluation Metrics ===")
    compute_traditional_metrics(generated_answers, actual_answers, args.model_type)
    
    # If CSV file is provided, compute conditional BERTScore
    if args.csv_file:
        print("\n=== Conditional BERTScore Evaluation ===")
        compute_conditional_bertscore(args.csv_file, retrieved_context_ids, args.model_type)

if __name__ == "__main__":
    main()
