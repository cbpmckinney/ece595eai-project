import json
import pandas as pd

INPUT_JSON = "llama4results.json"
OUTPUT_XLSX = "llama4results.xlsx"

# 1. Load JSON
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

# 2. Build DataFrame

df = pd.DataFrame(data)

# 3. Group by category + role and write each combo to its own sheet
EXCEL_COLS = ["id", "prompt-response", "followup-response"]

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    for (category, role), group in df.groupby(["category", "role"]):
        sheet_name = f"{category}_{role}"
        
        group[EXCEL_COLS].to_excel(writer, sheet_name=sheet_name, index=False)

        
print(f"Wrote {OUTPUT_XLSX}")