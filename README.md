# Text Feeder
A minimal code base to generate synthetic data using OpenAI to improve text classification models

## Parameter tbd
- `pip install openai langchain`
- under `params.py`
  - Change `GPT_MODEL`, `OPENAI_API_KEY`
  - Change `OUTPUT_FILE_PATH`
  - Choose `task` and change `kind_of_text`,`Labels`, `N_of_samples` and the example. 
  - potentially adapt `prompt_template`

## Run Tool
- Run with `python main.py`