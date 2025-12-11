import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from itertools import product, combinations
import pickle
import numpy as np


INPUT_JSON = "llama4results.json"
OUTPUT_JSON = "llama4results-sanitized.json"


# 1. Load JSON
with open(INPUT_JSON, "r") as f:
    data = json.load(f)
# 2. Build DataFrame
df = pd.DataFrame(data)
# 3. Organize data
categories = ["hobbies", "toys", "careers", "academics"]
genders = ["male", "female", "child"]
roles = ["none", "educator"]
print(repr(df["prompt-response"].iloc[0]))

df["prompt-response"] = df["prompt-response"].str.replace(
    r'(?s)^.*?\(1\)', '(1)', regex=True
)
print(repr(df["prompt-response"].iloc[0]))
df.to_json(OUTPUT_JSON, orient="records", indent=2, force_ascii=False)