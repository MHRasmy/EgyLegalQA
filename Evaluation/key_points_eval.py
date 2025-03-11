import argparse
import time
import pandas as pd
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

def configure_api(api_key):
    """Configure the API key for Gemini models."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.0-pro')

def load_data(file_path):
    """Load dataset from the given file path."""
    return pd.read_csv(file_path)

def evaluate_models(data, models, model_api, delay):
    """Evaluate multiple models and store the results."""
    for model in models:
        eval_column = f'{model}eval'
        data[eval_column] = None # Create a new column for evaluation results of each model
        for index, row in data.iterrows():
            question = row['question']
            key_points = "، ".join(row['key_points']) if isinstance(row['key_points'], list) else row['key_points']
            generated_answer = row[f'{model}rag']
            print(index)
            print(question)
            print(key_points)
            print(generated_answer)
            prompt = f"""
            In this task, you will receive a question, a generated answer, and a key point from a standard answer. Identify the part of the generated answer that most closely matches the key point of the standard answer; this will be referred to as the "comparable segment" of the generated answer. Analyze whether the comparable segment is relevant and consistent with, unrelated to, or contradictory to the key point of the standard answer. Provide a brief explanation of your analysis and conclude with one of the following labels:

            [[[Relevant]]]: Indicates that a segment of the generated answer is related to and consistent with the key points described in the reference answer. Note that only the essential information required to answer the question needs to be consistent; not all details are necessary. Each key point in the reference answer is generally essential, though some details may not be critical. When making a judgment, check if the most important information is included to consider it as relevant to the key points of the reference answer.

            [[[Irrelevant]]]: Indicates that the compared segment of the generated response does not contain or involve the relevant parts of the key points in the standard answer or that there is insufficient information in the generated answer.

            [[[Wrong]]]: Indicates that the compared segment of the generated response is incorrect based on the key points in the standard answer on major issues. If the generated answer states that there is not enough information to answer a certain part, count it as "Irrelevant" rather than "Wrong." Only when the generated answer gives a specific, incorrect answer should it be counted as "Wrong." Minor details that do not impact the answer to the question should be considered "Irrelevant," not "Wrong."

            Each request will have only one key point, so you can only output one of [[[Relevant]]], [[[Irrelevant]]], or [[[Wrong]]]. Comprehensively understand the question and analyze whether the comparable segment of the generated answer is relevant and consistent with, unrelated to, or contradictory to the information in the key points. Briefly explain your analysis in the key point evaluation, then provide the final conclusion.

            ---

            **Example 1:**

            **Question:** ما هو حكم نفقة المطلقة التي تستحق النفقة في القانون المصري؟
            **Generated Answer:** لا تستحق المطلقة النفقة من زوجها السابق بعد الطلاق إلا في حالات استثنائية.
            **Standard Answer Key Point:** 
            1. نفقة المطلقة تعتبر دَيناً في ذمة الزوج.
            **Key Point Evaluation:**
            - **Comparable Segment in Generated Answer:** "لا تستحق المطلقة النفقة من زوجها السابق بعد الطلاق إلا في حالات استثنائية"
            - **Analysis:** The generated answer contradicts the key point, which states that the wife is entitled to support as a debt on the husband.
            - **Conclusion:** [[[Wrong]]]

            ---

            **Example 2:**

            **Question:** متى تجب نفقة الزوجة على زوجها بموجب القانون المصري؟
            **Generated Answer:** يجب على الزوج النفقة على زوجته بعد الزواج مباشرة.
            **Standard Answer Key Point:** 
            1. نفقة الزوجة تجب من تاريخ العقد الصحيح.\n2. النفقة تُستحق إذا سلّمت الزوجة نفسها للزوج.
            **Key Point Evaluation:**
            - **Comparable Segment in Generated Answer:** "يجب على الزوج النفقة على زوجته بعد الزواج مباشرة"
            - **Analysis:** The generated answer provides an accurate but simplified version of the key point; it aligns with the requirement to support after marriage but omits the detail about submission.
            - **Conclusion:** [[[Relevant]]]

            ---

            **Example 3:**

            **Question:** ماذا يحدث إذا طلق القاضي الزوج بسبب عدم الإنفاق ثم أثبت الزوج يساره ورغبته في الرجوع لزوجته؟
            **Generated Answer:** يتم طلاق الزوجة إذا طلبت ذلك ولم ينفق الزوج.
            **Standard Answer Key Point:** 
            1. يحق للزوج الرجوع إلى زوجته.\n2. يجب على الزوج إثبات قدرته على الإنفاق لاستعادة حقوقه.
            **Key Point Evaluation:**
            - **Comparable Segment in Generated Answer:** "يتم طلاق الزوجة إذا طلبت ذلك ولم ينفق الزوج"
            - **Analysis:** The generated answer does not mention the right of the husband to return to the wife upon proving his ability to support her, which is a major part of the key point.
            - **Conclusion:** [[[Irrelevant]]]

            ---

            **Example 4:**

            **Question:** ما حكم زواج زوجة المفقود إذا ظهر حياً؟
            **Generated Answer:** يبطل زواج زوجة المفقود في حال عودته.
            **Standard Answer Key Point:** 
            1. زوجة المفقود تعود إليه عند إثبات أنه حي.\n2. الزواج الثاني يعتبر باطلاً قانونياً.'
            **Key Point Evaluation:**
            - **Comparable Segment in Generated Answer:** "يبطل زواج زوجة المفقود في حال عودته"
            - **Analysis:** The generated answer is consistent with the key point, as it accurately mentions the invalidation of the second marriage if the husband returns.
            - **Conclusion:** [[[Relevant]]]

            ---

            **Example 5:**

            **Question:** ما الإجراءات التي تتخذها المحكمة إذا امتنع الزوج عن الإنفاق على زوجته؟
            **Generated Answer:** المحكمة قد تصدر أمرًا بإنفاق الزوج على زوجته إذا ثبت امتناعه.
            **Standard Answer Key Point:** 
            1. المحكمة يمكنها إصدار أمر لتنفيذ النفقة جبراً.\n2. الإجراءات تتخذ عندما يثبت امتناع الزوج عن الإنفاق.
            **Key Point Evaluation:**
            - **Comparable Segment in Generated Answer:** "المحكمة قد تصدر أمرًا بإنفاق الزوج على زوجته إذا ثبت امتناعه"
            - **Analysis:** The generated answer is relevant as it aligns with the key point about the court’s authority to order support enforcement.
            - **Conclusion:** [[[Relevant]]]

            ---

            Example 6:
            **Question**: ماذا يحدث إذا طلق القاضي الزوج بسبب عدم الإنفاق ثم أثبت الزوج يساره ورغبته في الرجوع لزوجته؟
            **Generated Answer**: 
            يمكن للزوج ان يعود الى زوجته اذا استطاع الاثبات انه كان ذو قدرة مالية على الانفاق قبل الطلاق و يستطيع تحمل اعباء الحياة الزوجية مرة اخرى
            **Standard Answer Key Point**:
            1. يحق للزوج الرجوع إلى زوجته.\n2. يجب على الزوج إثبات قدرته على الإنفاق لاستعادة حقوقه.
            **Key Point Evaluation**:
            - **Comparable Segment in Generated Answer:** "يمكن للزوج ان يعود الى زوجته اذا استطاع الاثبات انه كان ذو قدرة مالية على الانفاق قبل الطلاق و يستطيع تحمل اعباء الحياة الزوجية مرة اخرى" 
            - **Analysis:** The generated answer aligns with the key points by acknowledging the husband's right to return to his wife upon proving his financial ability.  
            - **Conclusion:** [[[Relevant]]]

            ---

            Example 7:
            **Question**: ماذا يحدث إذا طلق القاضي الزوج بسبب عدم الإنفاق ثم أثبت الزوج يساره ورغبته في الرجوع لزوجته؟
            **Generated Answer**: 
            إذا كان الطلاق قد وقع من قبل
            **Standard Answer Key Point**:
            1. يحق للزوج الرجوع إلى زوجته.\n2. يجب على الزوج إثبات قدرته على الإنفاق لاستعادة حقوقه.
            **Key Point Evaluation**:
            - **Comparable Segment in Generated Answer:** "إذا كان الطلاق قد وقع من قبل". 
            - **Analysis:** The generated answer is vague and does not address the main points of the standard answer. It neither confirms the husband’s right to return to his wife nor mentions the need for proving his financial ability to support her. The response lacks critical information required by the key points.  
            - **Conclusion:** [[[Irrelevant]]]

            ---

            Test cases:
            Question: {question}
            Generated Answer: {generated_answer}
            Standard Answer Key Point: {key_points}
            Key Point Evaluation:
            """

            try:
                response = model_api.generate_content(prompt)
            except ResourceExhausted:
                print("Resource exhausted. Sleeping for 1 hour.")
                time.sleep(3600)
                response = model_api.generate_content(prompt)

            response_text = extract_response(response)
            data.at[index, eval_column] = response_text
            
            time.sleep(delay)  
    return data

def extract_response(response):
    """Extract and classify response from the model."""
    try:
        if response.parts:
            response_text = response.parts[0].text
        else:
            return None
    except AttributeError:
        return None
    
    if "[[[Relevant]]]" in response_text:
        return "Relevant"
    elif "[[[Irrelevant]]]" in response_text:
        return "Irrelevant"
    elif "[[[Wrong]]]" in response_text:
        return "Wrong"
    return "UNKNOWN"

def print_evaluation_results(data, models):
    """Print evaluation statistics for each model."""
    for model in models:
        model_eval_column = f'{model}eval'
        print(f"Model: {model}")
        print(data[model_eval_column].value_counts())

        eval_list = data[model_eval_column].dropna().tolist()
        wrong_count = sum("Wrong" in resp for resp in eval_list)
        relevant_count = sum("Relevant" in resp for resp in eval_list)
        irrelevant_count = sum("Irrelevant" in resp for resp in eval_list)
        total = len(eval_list)

        print(f"Hallu Ratio: {wrong_count / total if total > 0 else 0}")
        print(f"Completeness Ratio: {relevant_count / total if total > 0 else 0}")
        print(f"Irrelevant Ratio: {irrelevant_count / total if total > 0 else 0}\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate multiple models on a dataset")
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset CSV file')
    parser.add_argument('--models', type=str, nargs='+', required=True, help='List of models to evaluate')
    parser.add_argument('--api_key', type=str, required=True, help='Gemini API Key')
    parser.add_argument('--delay', type=int, default=20, help='Delay between API calls (default: 20s)')
    
    args = parser.parse_args()
    
    model_api = configure_api(args.api_key)
    data = load_data(args.dataset)
    data = evaluate_models(data, args.models, model_api, args.delay)
    print_evaluation_results(data, args.models)
    data.to_csv("evaluated_results.csv", index=False)
    print("Evaluation complete. Results saved to evaluated_results.csv")

if __name__ == "__main__":
    main()