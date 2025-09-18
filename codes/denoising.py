import torch
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizer
import numpy as np
from sklearn.metrics import mean_squared_error, confusion_matrix
from scipy.stats import spearmanr
import torch.nn as nn

Review_Rank_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", 
    num_labels=1
)

def compute_metrics(eval_pred, threshold=5):
    logits, labels = eval_pred
    mse = mean_squared_error(labels, logits)
    spearman_corr, _ = spearmanr(labels, logits)
    
    pred_class = (np.array(logits) >= threshold)
    labels_class = (np.array(labels) >= threshold)
    cm = confusion_matrix(labels_class, pred_class)
    return {
        "mse": mse,
        "spearman": spearman_corr,
        "confusion_matrix": cm.tolist()
    }

batch = [
  {"input_ids": [101, 2023, 2003, 102], "attention_mask": [1,1,1,1], "labels": 4.7, "confidence": 0.9},
  {"input_ids": [101, 2054, 2003, 102], "attention_mask": [1,1,1,1], "labels": 2.3, "confidence": 0.7},
  {"input_ids": [101, 2023, 2003, 102], "attention_mask": [1,1,1,1], "labels": 4.7, "confidence": 0.9},
  {"input_ids": [101, 2054, 2003, 102], "attention_mask": [1,1,1,1], "labels": 2.3, "confidence": 0.7},
]
# example of dataset after gemini labelling

def data_tensor_fn(batch):
    input_ids = torch.tensor([x["input_ids"] for x in batch])
    attention_mask = torch.tensor([x["attention_mask"] for x in batch])
    labels = torch.tensor([x["labels"] for x in batch], dtype=torch.float)
    weights = torch.tensor([x.get("confidence", 1.0) for x in batch], dtype=torch.float)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "weights": weights}

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')  # keep per-sample losses

    def forward(self, labels, outputs, weights=None):
        loss = self.mse(outputs.view(-1), labels)
        if weights is not None:
            loss = loss * weights
        return loss.mean()
    
def compute_loss(Review_Rank_model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    weights = inputs.pop("weights", None)
    outputs = Review_Rank_model(**inputs)
    logits = outputs.logits.view(-1)
    loss_fn = WeightedMSELoss()
    loss = loss_fn(labels, logits, weights)
    return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir="./results/denoise_bert",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,             
    per_device_train_batch_size=128,
    per_device_eval_batch_size=64,
    num_train_epochs=25,
    gradient_accumulation_steps=1,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_strategy="epoch",
    fp16=True,
    gradient_clip_val=1.0,
    lr_scheduler_type="linear",      #You could experiment with cosine if you notice training loss drops 
    warmup_ratio=0.1                 #quickly but then plateaus too early.
)

trainer = Trainer(
    model=Review_Rank_model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    compute_metrics=compute_metrics,
    tokenizer=DistilBertTokenizer,
    data_collator=data_tensor_fn,
    compute_loss=compute_loss
)