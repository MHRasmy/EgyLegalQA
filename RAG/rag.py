#!/usr/bin/env python
import argparse
import os
import pickle
from pymilvus import connections, utility, Collection, FieldSchema, CollectionSchema, DataType
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import nltk

# Set environment variable for CUDA allocation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('punkt')

def parse_args():
    parser = argparse.ArgumentParser(description="RAG Retrieval and Generation Pipeline")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to CSV dataset")
    parser.add_argument('--context_column', type=str, default="Chunk", help="Column name for context")
    parser.add_argument('--question_column', type=str, default="question", help="Column name for question")
    parser.add_argument('--answer_column', type=str, default="answer", help="Column name for answer")
    parser.add_argument('--milvus_uri', type=str, default="./egylaw_squad_dataset.db", help="Milvus connection URI")
    parser.add_argument('--collection_name', type=str, default="egylaw_squad_dataset", help="Name of Milvus collection")
    parser.add_argument('--embedding_model_name', type=str, default="intfloat/multilingual-e5-large", help="SentenceTransformer model name")
    parser.add_argument('--hf_token', type=str, required=True, help="Huggingface token for login")
    parser.add_argument('--lm_model_name', type=str, required=True, help="Language model for generation (e.g., 'gpt2')")
    parser.add_argument('--max_new_tokens', type=int, default=2000, help="Max new tokens for generation")
    parser.add_argument('--top_k', type=int, default=1, help="Top k for Milvus search")
    parser.add_argument('--output_pickle', type=str, default="retrieved_data.pkl", help="Output pickle file path")
    return parser.parse_args()

def login_hf(token):
    login(token=token)

def load_and_preprocess_dataset(dataset_path, question_col, answer_col):
    dataset = load_dataset('csv', data_files=dataset_path)
    def filter_null_rows(example):
        return example[question_col] is not None and example[answer_col] is not None
    dataset = dataset.filter(filter_null_rows)
    def add_index_column(examples, idx):
        examples["id"] = idx
        return examples
    dataset = dataset.map(add_index_column, with_indices=True)
    return dataset

def connect_milvus(milvus_uri):
    try:
        connections.connect(uri=milvus_uri)
        print("Connected to Milvus.")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        raise

def create_or_get_collection(collection_name, embedding_dim):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1000),
        FieldSchema(name="question_embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
        FieldSchema(name="context", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="context_embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
        FieldSchema(name="answers", dtype=DataType.VARCHAR, max_length=4000)
    ]
    schema = CollectionSchema(fields, description="Dataset embeddings")
    if not utility.has_collection(collection_name):
        collection = Collection(name=collection_name, schema=schema)
    else:
        collection = Collection(name=collection_name)
    return collection

def embed_and_insert_dataset(dataset, embedding_model, collection, context_col, question_col, answer_col):
    ids, questions, question_embeddings = [], [], []
    contexts, context_embeddings, answers = [], [], []
    for item in tqdm(dataset['train'], desc="Embedding dataset"):
        ids.append(int(item['id']))
        questions.append(item[question_col])
        question_embeddings.append(embedding_model.encode(item[question_col]).tolist())
        contexts.append(item[context_col])
        context_embeddings.append(embedding_model.encode(item[context_col]).tolist())
        answers.append(item[answer_col])
    collection.insert([ids, questions, question_embeddings, contexts, context_embeddings, answers])
    print(f"Embedded data inserted into Milvus collection: {collection.name}")
    collection.create_index("question_embedding", {"index_type": "HNSW", "metric_type": "COSINE"})
    collection.create_index("context_embedding", {"index_type": "HNSW", "metric_type": "COSINE"})
    collection.load()
    print(f"Number of entities in the collection: {collection.num_entities}")

def generate_prompt(question, context):
    return f"""
        انت مساعد مفيد متخصص في الاجابة عن الاستشارات القانونية في الاحوال الشخصية في القانون المصري. 
        مهمتك هي الاجابة عن السؤال التالي: '{question}' 
        في ضوء القطعة التالية: '{context}'
        
        الاجابة هي: 
    """

def main():
    args = parse_args()
    login_hf(args.hf_token)
    dataset = load_and_preprocess_dataset(args.dataset_path, args.question_column, args.answer_column)
    connect_milvus(args.milvus_uri)
    embedding_model = SentenceTransformer(args.embedding_model_name)
    embedding_dim = embedding_model.get_sentence_embedding_dimension()
    collection = create_or_get_collection(args.collection_name, embedding_dim)
    embed_and_insert_dataset(dataset, embedding_model, collection, args.context_column, args.question_column, args.answer_column)
    
    # Setup language model for generation
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model_name)
    model = AutoModelForCausalLM.from_pretrained(args.lm_model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    actual_context_ids, retrieved_context_ids = [], []
    actual_answers, expected_retrieved_answers, generated_answers = [], [], []
    
    for item in tqdm(dataset['train'], desc="Retrieval and Generation"):
        question_id = item['id']
        question_text = item[args.question_column]
        actual_answer = item[args.answer_column]
        
        actual_context_ids.append(question_id)
        actual_answers.append(actual_answer)
        
        # Retrieve question embedding from Milvus
        query_result = collection.query(expr=f"id == {question_id}", output_fields=["question_embedding"])
        question_embedding = query_result[0]['question_embedding']
        
        # Search for the closest context
        search_params = {"metric_type": "COSINE"}
        results = collection.search(data=[question_embedding],
                                    anns_field="context_embedding",
                                    param=search_params,
                                    limit=args.top_k,
                                    output_fields=["context", "answers"])
        closest_context = results[0][0].context
        retrieved_context_id = results[0][0].id
        expected_answer = results[0][0].answers
        
        retrieved_context_ids.append(retrieved_context_id)
        expected_retrieved_answers.append(expected_answer)
        
        prompt = generate_prompt(question_text, closest_context)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        gen_ids = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
        response = tokenizer.decode(gen_ids[0], skip_special_tokens=True).split("الاجابة هي:")[-1].strip()
        generated_answers.append(response)
        print(f"Processed question id: {question_id}")
        torch.cuda.empty_cache()
    
    with open(args.output_pickle, "wb") as f:
        pickle.dump({
            'actual_context_ids': actual_context_ids,
            'retrieved_context_ids': retrieved_context_ids,
            'actual_answers_of_queried_questions': actual_answers,
            'expected_retrieved_answers': expected_retrieved_answers,
            'generated_answers': generated_answers
        }, f)
    print(f"Data saved to {args.output_pickle}")

if __name__ == "__main__":
    main()
