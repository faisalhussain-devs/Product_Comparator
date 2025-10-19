import json
from pathlib import Path
from datasets import Dataset
from setfit import SetFitModel, SetFitTrainer
import torch
import numpy as np
import sys

# === Paths ===
DATA_PATH = Path(r"E:\Product Comparator\aspect_analysis\final_aspa_data_cleaned.json")
MODEL_PATH = Path(r"E:\Product Comparator\model_aspa")
SAVE_PATH = Path(r"E:\Product Comparator\model_aspa_full")

# === Load dataset ===
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list([{"text": d["input"], "label": d["label"]} for d in data])

# === Encode labels as multi-hot vectors ===
all_labels = sorted({lbl for d in data for lbl in d["label"]})
label2id = {l: i for i, l in enumerate(all_labels)}
id2label = {i: l for l, i in label2id.items()}

def encode_labels(example):
    y = [0] * len(all_labels)
    for l in example["label"]:
        if l in label2id:
            y[label2id[l]] = 1
    example["label"] = y
    return example

dataset = dataset.map(encode_labels, batched=False, disable_nullable=True)

# === Shuffle and split ===
dataset = dataset.shuffle(seed=42)
train_ds, test_ds = dataset.train_test_split(test_size=0.1).values()

# === Load backbone ===
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()
model = SetFitModel.from_pretrained(MODEL_PATH).to(device)
print("✅ Loaded backbone from checkpoint.\n")

# === Precompute embeddings to save RAM ===
print("🔄 Precomputing embeddings for train and test sets...")
X_train = model.encode(train_ds["text"], device=device)
y_train = np.array(train_ds["label"])

X_test = model.encode(test_ds["text"], device=device)
y_test = np.array(test_ds["label"])
print("✅ Embeddings precomputed.\n")

# === Use a lightweight sklearn head for training ===
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Create binary classifier for multi-label
head = SGDClassifier(loss="log", max_iter=1000, tol=1e-3, random_state=42)
print("🚀 Training classification head only...")
head.fit(X_train, y_train)
print("✅ Head training finished.\n")

# === Evaluate ===
y_pred = head.predict(X_test)
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

print("🔍 Evaluation metrics:")
print("F1 (micro):", f1_score(y_test, y_pred, average="micro"))
print("Precision (micro):", precision_score(y_test, y_pred, average="micro"))
print("Recall (micro):", recall_score(y_test, y_pred, average="micro"))
print("\nDetailed report:\n")
print(classification_report(y_test, y_pred, target_names=all_labels, zero_division=0))

# === Save backbone + head ===
import joblib
model.save_pretrained(SAVE_PATH)
joblib.dump(head, SAVE_PATH / "classifier_head.pkl")
print(f"✅ Model and head saved to {SAVE_PATH}\n")

# === Inference example ===
new_review = "Coming from an iPhone, the battery lasts longer and display looks brighter."
emb = model.encode([new_review], device=device)
preds = head.predict(emb)[0]
predicted_labels = [id2label[i] for i, v in enumerate(preds) if v == 1]
print("Predicted aspects:", predicted_labels)

sys.stdout.flush()
