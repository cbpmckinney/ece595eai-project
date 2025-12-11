import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from itertools import product, combinations
import pickle
import numpy as np


INPUT_JSON = "llama4results-sanitized.json"


# 1. Load JSON
with open(INPUT_JSON, "r") as f:
    data = json.load(f)
# 2. Build DataFrame
df = pd.DataFrame(data)
# 3. Organize data
categories = ["hobbies", "toys", "careers", "academics"]
genders = ["male", "female", "child"]
roles = ["none", "educator"]

frames = {}

for c, g, r in product(categories, genders, roles):
    key = f"{c}_{g}_{r}"
    subset = df[(df["category"] == c) & (df["gender"] == g) & (df["role"] == r)].copy()
    subset = subset.reset_index(drop=True)
    frames[key] = subset 

#print(frames["hobbies_male_none"].iloc[0]["prompt-response"])
#print(frames["hobbies_female_none"].iloc[0]["prompt-response"])

# Set up embedding model
model = SentenceTransformer("all-mpnet-base-v2")




for c, g, r in product(categories, genders, roles):
    key = f"{c}_{g}_{r}"
    response_text = frames[key]["prompt-response"].tolist()
    response_embedding = model.encode(response_text, batch_size=32, convert_to_numpy=True)
    followup_text = frames[key]["followup-response"].tolist()
    followup_embedding = model.encode(followup_text, batch_size=32, convert_to_numpy=True)
    frames[key]["response-embedding"] = list(response_embedding)
    frames[key]["followup-embedding"] = list(followup_embedding)



def cosine_distance(v1, v2):
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return np.nan
    cos_sim=np.dot(v1, v2)/denom
    return 1.0 - cos_sim




newframes = {}
for c in categories:
    for g1, g2 in combinations(genders, 2):
        for r in roles:
            newkey = f"{c}_{g1}_{g2}_{r}"
            key1 = f"{c}_{g1}_{r}"
            key2 = f"{c}_{g2}_{r}"
            df1 = frames.get(key1)
            df2 = frames.get(key2)
            embs1 = df1["response-embedding"].iloc[:len(df1)]
            embs2 = df2["response-embedding"].iloc[:len(df2)]
            distances1 = [cosine_distance(e1, e2) for e1, e2 in zip(embs1, embs2)]
            ages = df1["age"].iloc[:len(df1)]
            
            embs3 = df1["followup-embedding"].iloc[:len(df1)]
            embs4 = df2["followup-embedding"].iloc[:len(df2)]
            distances2 = [cosine_distance(e1, e2) for e1, e2 in zip(embs3, embs4)]
            
            response_diffs = [e1 - e2 for e1, e2 in zip(embs1, embs2)]
            followup_diffs = [e1 - e2 for e1, e2 in zip(embs3, embs4)]
            
            
            newframes[newkey] = pd.DataFrame({
                "age": ages,
                "response-cosine-distance": distances1,
                "followup-cosine-distance": distances2,
                "response-diffvector": response_diffs,
                "followup-diffvector": followup_diffs
            })

            
# Save frames for use in ipynb

with open("frames-llama.pkl", "wb") as f:
    pickle.dump(frames, f)

with open("newframes-llama.pkl", "wb") as f:
    pickle.dump(newframes, f)
