## OpenAI Details
OPENAI_MODEL = "gpt-4-0125-preview"
OPENAI_API_KEY = "YOUR_API_KEY"

temp=0.7

## Labels and desired number of texts
labels = [
    ("Volleyball", 1),
    ("Skateboarding", 1),
    ("Soccer", 1),
    ("Tennis", 1),
    ("Golf", 1)
    ]

## Kind of text (eg. Tweet, project descrition, report, review etc.)
kind_of_text = "Tweet" 

## Example 
example_label = "Skateboarding"
example_text = "Finaly nailed that kickflip! So stoked!"

## Adapt prompt template
prompt_template = f"""
The goal is to generate a {kind_of_text} that fits the given label.
Here is an example for the label {example_label}: {example_text}
Important: Generate one {kind_of_text} per label.
"""

## Output Path
PATH = "synthetic_data.csv"