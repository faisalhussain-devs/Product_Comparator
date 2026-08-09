import json, re
from pathlib import Path

INPUT_PATH = Path(r"E:\Product Comparator\final_aspa_data_with_sentiments.json")
CLEANED_PATH = Path(r"E:\Product Comparator\final_aspa_data_cleaned_with_sentiments.json")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

def clean_label(label):
    label = label.strip()
    label = re.sub(r"\s+", " ", label)
    label = label[0].upper() + label[1:] if label else label
    return label

for sample in data:
    sample["sentiments"] = {clean_label(lbl): sentiment for lbl, sentiment in sample["sentiments"].items()}

with open(CLEANED_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f" Cleaned labels and saved to {CLEANED_PATH}")
