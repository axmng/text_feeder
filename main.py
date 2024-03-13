import openai
import pandas as pd
from params import OPENAI_API_KEY, OPENAI_MODEL, PATH, temp, labels, prompt_template
from langchain import PromptTemplate, OpenAI
from openai import OpenAI

client = OpenAI(api_key=OPENAI_API_KEY)
# Function to generate project descriptions for a given label
def infer_text(label):
    prompt = prompt_template + f"\nLabel: {label}\n"
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

# Loop through each category and generate project descriptions
for label, count in labels:
    for _ in range(count):
        text = infer_text(label)
        df.loc[len(df)] = [label, text]

# save as csv
df.to_csv(PATH, sep=";", index=False, encoding='utf-8-sig')