import openai
import pandas as pd
import json

# Import NER-related variables from params.py
from params import OPENAI_API_KEY, OPENAI_MODEL, PATH, temp, task, labels, prompt_template, prompt_template_ner, N_of_samples, example_ner, example_classi

from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)

def infer_text(label, last_text=None):
    example_json = json.dumps(example_classi)
    variability_prompt = "" if last_text is None else f"Make sure that the response is substantially different to the last one: {last_text}"
    prompt = prompt_template + f"\nExamples:\n{example_json}\n{variability_prompt}\nLabel: {label}\nResponse is expected in JSON format."
    
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful AI that creates synthetic data for machine learning tasks."},
            {"role": "user", "content": prompt}
        ],
        temperature=temp,
    )
    response_content = response.choices[0].message.content.replace("```json\n", "").replace("```", "")
    try:
        response_json = json.loads(response_content)
    except json.JSONDecodeError as e:
        print("JSON parsing error:", e)
        print("Raw response content:", response_content)
        return None

    return response_json[0]['text']

def infer_ner():
    example_json = json.dumps(example_ner)
    prompt = prompt_template_ner + f"\nExample:\n{example_json}\nResponse is expected in JSON format."

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful AI that generates synthetic newspaper headlines for NER tasks."},
            {"role": "user", "content": prompt}
        ],
        temperature=temp,
    )
    response_content = response.choices[0].message.content.replace("```json\n", "").replace("```", "")
    try:
        response_json = json.loads(response_content)
    except json.JSONDecodeError as e:
        print("JSON parsing error:", e)
        print("Raw response content:", response_content)
        return None

    return response_json

if task == "text classification":
    df = pd.DataFrame(columns=['label', 'text'])
    last_texts = {}
    
    for label in labels:
        last_text = None
        for _ in range(N_of_samples):
            text = infer_text(label, last_text)
            df.loc[len(df)] = [label, text]
            last_text = text

    # Save the DataFrame as CSV for text classification
    df.to_csv(PATH, sep=";", index=False, encoding='utf-8-sig')
elif task == "NER":
    ner_data = []
    for _ in range(N_of_samples):
        ner_entry = infer_ner()
        if ner_entry:
            ner_data.append(ner_entry)
    
    # Convert the NER data list to a DataFrame and save it as CSV
    df_ner = pd.DataFrame(ner_data)
    df_ner.to_csv(PATH, sep=";", index=False, encoding='utf-8-sig')
