## OpenAI Details
OPENAI_MODEL = "gpt-4-0125-preview"
OPENAI_API_KEY = "API_KEY"

temp=0.7

task = "NER" ## "text classification" OR "NER"

## Kind of text (eg. Tweet, project descrition, report, review etc.)
kind_of_text = "newspaper headlines" 
# kind_of_text = "Tweet"

## Labels
labels = [
    "PERSON",
    "ORGANISATION",
    "LOCATION"
]
# labels = ["Volleyball", "Skateboarding", "Soccer", "Tennis", "Golf"]

## Specify 
N_of_samples = 2

## Output Path
PATH = "synthetic_data.csv"

## Text Classification
## Example for Text Classification
example_classi = [
  {
    "label": "Skateboarding",
    "text": "Finally nailed that kickflip! So stoked!"
  }
]

## Prompt template for Text Classification
prompt_template = f"""
The goal is to generate a {kind_of_text} that fits the given label.
Important: Generate one {kind_of_text} per label. Response is expected in JSON format.
"""

## NER
## Example for NER
example_ner = [
  {
    "text": "Ralph Wiggum to Star in New Springfield Symphony Orchestra Series, Mayor Announces at City Hall",
    "ner": {
      "PERSON": ["Ralph Wiggum", "Mayor"],
      "ORGANIZATION": ["Springfield Symphony Orchestra"],
      "LOCATION": ["City Hall"]
    }
  }
]

## Prompt template for NER
prompt_template_ner = f"""
The goal is to generate {kind_of_text} that can be used for Named Entity Recognition (NER). The {kind_of_text} should at least contain one this labels: {labels}. Once the text is generated please also provide the correct labeled entities.
Important: Only generate one {kind_of_text}. Response is expected in JSON format.
"""