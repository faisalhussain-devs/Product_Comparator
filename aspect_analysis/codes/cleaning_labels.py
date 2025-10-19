import json, re
from pathlib import Path

INPUT_PATH = Path(r"E:\Product Comparator\aspect_analysis\final_aspa_data.json")
CLEANED_PATH = Path(r"E:\Product Comparator\aspect_analysis\final_aspa_data_cleaned.json")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

def clean_label(label):
    label = label.strip()
    label = re.sub(r"\s+", " ", label)
    label = label[0].upper() + label[1:] if label else label
    return label

for sample in data:
    sample["label"] = sorted(list(set(clean_label(lbl) for lbl in sample["label"])))

with open(CLEANED_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f" Cleaned labels and saved to {CLEANED_PATH}")
