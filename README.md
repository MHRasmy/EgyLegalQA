# Egyptian Legal QA System

This repository contains an LLM-powered Retrieval-Augmented Generation (RAG) system and evaluation tools for a legal question answering (QA) application focused on Egyptian personal status law. The project is organized into three main folders:

- **[RAG](./RAG/rag.py):** Contains the code for data embedding, retrieval, and answer generation using Milvus, SentenceTransformers, and a language model.
- **[Evaluation](./Evaluation/qa_eval.py) & [Key Points Evaluation](./Evaluation/key_points_eval.py):** Contains scripts for evaluating generated answers with metrics such as BERTScore, BLEU, ROUGE-L, and Conditional BERTScore.
- **[QA Generation](./QA%20Generation/qa_gen.py):** Contains the code to generate question-answer pairs from semantically segmented legal contexts.

---

## Setup

Install all the required dependencies using the following command:

```bash
pip install pymilvus sentence-transformers bitsandbytes tqdm evaluate bert_score rouge-score nltk datasets transformers scikit-learn huggingface_hub google-generativeai --quiet
```

---

## Folder Details

### 1. RAG Retrieval & Generation

The RAG pipeline is implemented in the **[RAG/rag.py](./RAG/rag.py)** script. It:
- Loads a CSV dataset.
- Connects to a Milvus vector store.
- Embeds dataset texts using a SentenceTransformer model.
- Inserts data into a Milvus collection and builds indexes.
- Retrieves the most relevant context for each question.
- Generates answers using a specified language model.

**Usage:**

```bash
python RAG/rag.py --dataset_path path/to/your_dataset.csv \
                  --hf_token YOUR_HF_TOKEN \
                  --lm_model_name your-generation-model \
                  --embedding_model_name intfloat/multilingual-e5-large
```

Run `python RAG/rag.py --help` for more options.

---

### 2. QA Evaluation

The QA evaluation pipeline is implemented in **[Evaluation/qa_eval.py](./Evaluation/qa_eval.py)**. It computes:
- Traditional evaluation metrics (BERTScore, BLEU, ROUGE-L)
- **Conditional BERTScore** – which incorporates the context and question into the evaluation.

**Usage:**

```bash
python Evaluation/qa_eval.py --pickle_file retrieved_data.pkl \
                             --model_type distilbert-base-multilingual-cased \
                             --csv_file path/to/your_data.csv \
                             --context_col Chunk \
                             --question_col question \
                             --answer_col answer \
                             --generated_col jais_adapted_rag
```

If you wish to compute only traditional metrics, omit the `--csv_file` flag.

---

### 3. QA Generation

The QA Generation script is implemented in **[QA Generation/qa_gen.py](./QA%20Generation/qa_gen.py)**. It generates question-answer pairs from a CSV file containing segmented legal contexts. The script uses the Google Generative AI API to produce structured QA pairs.

**Usage:**

```bash
python "QA Generation/qa_gen.py" --dataset path/to/legal_chunks.csv \
                                 --api_key YOUR_API_KEY \
                                 --model_name gemini-1.0-pro-latest \
                                 --start_index 0 \
                                 --delay 20 \
                                 --output output_questions_answers.csv
```

Run `python "QA Generation/qa_gen.py" --help` for additional options.