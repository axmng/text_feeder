import openai
import pandas as pd
from params import OPENAI_API_KEY, OPENAI_MODEL, PATH, temp, labels, prompt_template, kind_of_text
from langchain import PromptTemplate, OpenAI
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)
# Function to generate project descriptions for a given label
def infer_text(label, last_text=None):
    variability_prompt = "" if last_text is None else f"Make sure that the {kind_of_text} is substantially different to the last {kind_of_text}: {last_text}"
    prompt = prompt_template + variability_prompt + f"\nLabel: {label}\n"
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
                {"role": "system", "content": "You are a helpful AI that creates synthetic data for machine learning tasks."},
                {"role": "user", "content": prompt}
            ],
        temperature=temp,
    )
    return response.choices[0].message.content

# Create an empty DataFrame to store the results
df = pd.DataFrame(columns=['label', 'text'])
last_texts = {}

# Loop through each category and generate project descriptions
for label, count in labels:
    last_text = None  # Initialize last_text as None for each label
    for _ in range(count):
        text = infer_text(label, last_text)
        df.loc[len(df)] = [label, text]
        last_text = text  # Update last_text with the newly generated text
        last_texts[label] = last_text 

# save as csv
df.to_csv(PATH, sep=";", index=False, encoding='utf-8-sig')