import openai
import csv
import json
import os
from dotenv import load_dotenv
import time
import random
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv('OPEN_AI_API'))

# List of diseases
disease_list = ["Anxiety", "Arthritis", "Cancer", "Chronic Disc Disease", "Collapsed Trachea", "Cushing's", "Degenerative Myelopathy","Diet and Food","Ear Infections","Gastric Disorders","Kidney Disease","Liver Disease","Neurological","Heart Disease","Pancreatitis","Skin Disorders","Ticks Fleas Heartworm","Vaccinations"]

# Analyze a batch of rows
def analyze_batch(batch, diseases):
    batch_text = ""
    for idx, row in enumerate(batch):
        batch_text += f"""
        Q{idx+1}: {row['question']}
        A{idx+1}: {row['answer']}
        """

    prompt = f"""
    Given the following question and answer pair, categorize the content into one or more diseases from the provided list ONLY, and provide a maximum of 5 highly relevant short keywords. Return a JSON object with 'diseases' and short 'keywords'.

    Diseases List: {', '.join(diseases)}

    Return a JSON array where each element corresponds to the same order of input pairs.
    Each element should be:
    {{"diseases": ["Disease1", "Disease2"], "keywords": ["shortkeyword1", "shortkeyword2", "shortkeyword3", "shortkeyword4", "shortkeyword5"]}}

    Question-Answer Pairs:
    {batch_text}
    """

    retries = 3
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Updated model name
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            result_json = json.loads(response.choices[0].message.content.strip())
            return result_json
        except Exception as e:
            wait_time = (2 ** attempt) + random.random()
            print(f"Error: {e}. Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)

    return [{} for _ in batch]  # fallback if all retries fail


# Process CSV in batches
def process_csv_in_batches(csv_file_path, diseases, batch_size=5):
    results = []

    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = list(csv.DictReader(csvfile))
        total_rows = len(reader)

        for start in range(0, total_rows, batch_size):
            end = start + batch_size
            batch = reader[start:end]
            print(f"Processing rows {start+1} to {min(end, total_rows)} of {total_rows}...", end="\r")

            analysis_results = analyze_batch(batch, diseases)

            for row_data, analysis in zip(batch, analysis_results):
                results.append({
                    "diseases": analysis.get("diseases", []),
                    "keywords": analysis.get("keywords", []),
                    "answer": row_data["answer"]
                })

            time.sleep(0.3)  # small delay to be safe

    # Save to JSON
    with open('analyzed_data.json', 'w', encoding='utf-8') as jsonfile:
        json.dump(results, jsonfile, ensure_ascii=False, indent=4)

    print("\nAnalysis completed and saved to analyzed_data.json")


# Run
csv_file_path = 'qa_dataset.csv'
process_csv_in_batches(csv_file_path, disease_list, batch_size=5)
