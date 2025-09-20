import json
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizer, AutoTokenizer
import numpy as np
from sklearn.metrics import mean_squared_error, confusion_matrix
from scipy.stats import spearmanr
import torch.nn as nn
import pandas as pd
from datasets import Dataset
from codes.labelling_data import labelling_data
def load_gemini_labeled_data(results_list, threshold=None):
    """
    Load and filter Gemini-labeled review data into a Hugging Face Dataset.

    This function takes a list of Gemini labeling results, converts it into a 
    pandas DataFrame, removes invalid rows, applies an optional usefulness score 
    threshold, and returns the cleaned dataset. It also prints useful statistics 
    about text lengths to select max_length.

    Args:
        results_list (list of dict): List of labeled review entries, where each 
            entry contains at least 'text' and 'usefulness_score' keys.
        threshold (float, optional): Minimum usefulness score required for a review 
            to be included. If None, no threshold filtering is applied.

    Returns:
        datasets.Dataset: A Hugging Face Dataset object.
    """

    df = pd.DataFrame(results_list) 
    df = df.dropna(subset=["text", "usefulness_score"])

    if threshold is not None:
        df = df[df["usefulness_score"] >= threshold]
    
    lengths = df["text"].apply(lambda x: len(x.split()))
    print("99th percentile length:", lengths.quantile(0.99))
    print("Max length:", lengths.max())
    print("Dataset size after filtering:", len(df))
    
    return Dataset.from_pandas(df, preserve_index=False)

def tokenize_and_collate(dataset, tokenizer, max_length=512):
    """
    Tokenize review texts and prepare a collated dataset for model training.

    Applies a Hugging Face tokenizer to review text, pads/truncates to a fixed 
    length, and attaches labels and sample weights. Labels come from the 
    'usefulness_score', while weights are derived from 'confidence_score' if 
    available, otherwise default to 1.0. The dataset is returned in PyTorch format.

    Args:
        dataset (datasets.Dataset): Input dataset with at least 'text' and 
            'usefulness_score' columns, and optionally 'confidence_score'.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding text.
        max_length (int, optional): Maximum sequence length for padding/truncation. 
            Defaults to 512.

    Returns:
        datasets.Dataset: Tokenized dataset with PyTorch tensors for:
            - 'input_ids'
            - 'attention_mask'
            - 'labels' (usefulness score)
            - 'weights' (confidence-based sample weights)
    """

    # here max_length is chosen as 512 token around 350 words, check the max length of reviews in your reviews text
    def tokenize_batch(batch, default_weight=1.0):
        tokenized = tokenizer(
            batch['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=max_length
        )
        tokenized['labels'] = batch['usefulness_score']
        tokenized['weights'] = [ 
            w if w is not None else default_weight
                for w in batch.get("confidence_score", [default_weight] * len(batch["text"]))]
        return tokenized
    
    tokenized_dataset = dataset.map(tokenize_batch, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels', 'weights'])
    return tokenized_dataset

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

class WeightedMSELoss(nn.Module):
    """
Weighted Mean Squared Error (MSE) loss for regression tasks.
Computes per-sample squared errors and applies optional sample-level weights before averaging.

Args:
    labels (torch.Tensor): Ground truth values, shape (batch_size,).
    outputs (torch.Tensor): Model predictions, shape (batch_size, 1).
    weights (torch.Tensor, optional): Sample-level weights, shape 
        (batch_size,). Defaults to None.

Returns:
    torch.Tensor: Scalar tensor representing the weighted mean squared error.
"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')  # keep per-sample losses

    def forward(self, labels, outputs, weights=None):
        loss = self.mse(outputs.view(-1), labels)
        if weights is not None:
            loss = loss * weights
        return loss.mean()
    
def compute_loss(Review_Rank_model, inputs, return_outputs=False):
    """
    Compute the weighted mean squared error (MSE) loss for the review ranking model.
    Parameters -----
    Review_Rank_model : torch.nn.Module
        The model to evaluate. It should return an object with a `.logits` attribute containing the predicted scores for each input sample.
    inputs : dict
        A batch of input tensors. Must include:
          - "labels": torch.Tensor of true usefulness scores.
          - "weights" (optional): torch.Tensor of per-sample confidence weights.
          - Other keys required by the model (e.g., "input_ids", "attention_mask").

    return_outputs : bool, default=False
        If True, return both the loss and the model outputs. If False, return only the loss.

    Returns -------
    torch.Tensor or (torch.Tensor, Any)
        - If return_outputs is False: scalar tensor representing the computed loss.
        - If return_outputs is True: tuple (loss, outputs) where `outputs` is the model's raw output object.
    - Uses `WeightedMSELoss`
    """
    labels = inputs["labels"]
    weights = inputs.get("weights", None)
    outputs = Review_Rank_model(**inputs)
    logits = outputs.logits.view(-1)
    loss_fn = WeightedMSELoss()
    loss = loss_fn(labels, logits, weights)
    return (loss, outputs) if return_outputs else loss

if __name__ == "__main__":
    Review_Rank_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=1
    )

    GEMINI_OUTPUT = "labeled_reviews_final.jsonl"
    results = []
    """with open(GEMINI_OUTPUT, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))

    threshold = results[-1]["threshold"]
    results = results[:-1]"""
    results, threshold = labelling_data()

    print("Threshold:", threshold)
    print("First review:", results[0])
    data = load_gemini_labeled_data(results_list=results, threshold=threshold)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    data = data.shuffle(seed=42)
    train_val = data.train_test_split(test_size=0.2, seed=42) 
    train_data = train_val["train"]
    val_data = train_val["test"]

    tokenized_train = tokenize_and_collate(train_data, tokenizer, max_length=512) # do experiments for 99% percentile length of text or max length
    tokenized_val   = tokenize_and_collate(val_data, tokenizer, max_length=512)

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
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=tokenize_and_collate,
        compute_loss=compute_loss
    )

    trainer.save_model("notebooks/result/final_model")
