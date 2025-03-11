#!/usr/bin/env python
import argparse
import time
import pandas as pd
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

def parse_args():
    parser = argparse.ArgumentParser(description="QA Generation Script")
    parser.add_argument('--dataset', type=str, required=True, help="Path to CSV file containing context data (must include a 'Chunk' column)")
    parser.add_argument('--start_index', type=int, default=0, help="Start index for processing rows")
    parser.add_argument('--api_key', type=str, required=True, help="Your Google Generative AI API key")
    parser.add_argument('--model_name', type=str, default="gemini-1.0-pro-latest", help="Name of the generative model to use")
    parser.add_argument('--delay', type=int, default=20, help="Delay between API calls (in seconds)")
    parser.add_argument('--output', type=str, default="output_questions_answers.csv", help="Output CSV filename")
    return parser.parse_args()

def extract_qa(response_text):
    if "###INSUFFICIENT CONTEXT###" in response_text:
        return None, None
    try:
        question_part = response_text.split("**سؤال:**")[1]
        question = question_part.split("**إجابة:**")[0].strip()
        answer_part = response_text.split("**إجابة:**")[1]
        answer = answer_part.split('---')[0].strip()
        return question, answer
    except (IndexError, AttributeError):
        return None, None

def main():
    args = parse_args()
    genai.configure(api_key=args.api_key)
    model = genai.GenerativeModel(args.model_name)
    
    data = pd.read_csv(args.dataset, encoding="utf-8")
    data.dropna(axis=0, inplace=True)
    data['question'] = None
    data['answer'] = None

    for index, chunk in data['Chunk'].iloc[args.start_index:].items():
        prompt = f"""
        You are an advanced language model proficient in Arabic and knowledgeable about Egyptian personal status law. Your task is to generate one comprehensive question and one corresponding answer based solely on the provided Arabic text. Each question and answer pair should be directly related to the content and accurately reflect the information presented. The answer should be clear, concise, and correct based on the given context.
        
        **Requirements:**
        
        1. **Language:** Both the question and the answer must be in Arabic.
        2. **Format:** The output should consist of a single question followed by its corresponding answer.
        3. **Coverage:** Ensure that the question covers a major or detailed topic from the provided text, including definitions, procedures, rights, and obligations under the Egyptian personal status law.
        4. **[IMPORTANT] Insufficient Context Handling:** 
           - If the provided context is too short, meaningless, insufficient or doesn't make sense to generate a meaningful question and answer, include the following flag in your response: `###INSUFFICIENT CONTEXT###`
        
        **Provided Context:**
        
        ```
        {chunk}
        ```
        
        **Output Format:**
        
        Present the question and answer in a structured and organized manner as shown below:
        
        ---
        
        **سؤال:**
        
        [Insert the generated question here]
        
        **إجابة:**
        
        [Insert the generated answer here]
        
        ---
        
        Or, in case of insufficient context:
        
        ---
        
        ###INSUFFICIENT CONTEXT###
        """
        try:
            response = model.generate_content(prompt)
        except ResourceExhausted:
            print("Resource exhausted. Sleeping for 1 hour...")
            time.sleep(3600)
            response = model.generate_content(prompt)
            
        try:
            if response.parts:
                response_text = response.parts[0].text
            else:
                response_text = "###INSUFFICIENT CONTEXT###"
        except AttributeError:
            data.at[index, 'question'] = None
            data.at[index, 'answer'] = None
            continue

        question, answer = extract_qa(response_text)
        data.at[index, 'question'] = question
        data.at[index, 'answer'] = answer
        if question and answer:
            print(f"Index {index}\nQuestion:\n{question}\nAnswer:\n{answer}\n")
        time.sleep(args.delay)
    
    data.to_csv(args.output, index=False, encoding='utf-8-sig')
    print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()
