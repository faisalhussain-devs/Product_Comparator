import torch
from torch import nn
import pytest
import pandas as pd
from datasets import Dataset
from codes.denoising import load_gemini_labeled_data, tokenize_and_collate, WeightedMSELoss, compute_loss
from transformers import DistilBertTokenizer
from types import SimpleNamespace
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def test_load_gemini_labeled_data_basic():
    """
    Test that `load_gemini_labeled_data` works correctly by verifying:
    - Rows with usefulness scores below the threshold are ignored
    - Rows with `None` in `text` or `usefulness_score` columns are dropped
    - The output is a Hugging Face `Dataset` object
    - The dataset contains the expected columns: ['text', 'usefulness_score', 'confidence_score']
    """

    results_list = [
        {"id": 1, "text": "Great product", "usefulness_score": 8.5, "confidence_score": 0.9},
        {"id": 2, "text": "Bad quality", "usefulness_score": 2.0, "confidence_score": 0.8},
        {"id": 3, "text": None, "usefulness_score": 5.0, "confidence_score": 0.7},   
        {"id": 4, "text": "Okay", "usefulness_score": None, "confidence_score": 0.6},
        {"id": 5, "text": "Great product", "usefulness_score": 8.5, "confidence_score": 0.9},
        {"id": 6, "text": "Bad quality", "usefulness_score": 2.0, "confidence_score": 0.8},
        {"id": 7, "text": "Average", "usefulness_score": 5.0, "confidence_score": 0.7},
    ]
    dataset = load_gemini_labeled_data(results_list, threshold=5.0)
    assert len(dataset) == 3
    for row in dataset:
        assert row["score"] >= 5.0
    assert isinstance(dataset, Dataset)
    for col in ["text", "usefulness_score", "confidence_score"]:
        assert col in dataset.column_names



def test_tokenize_and_collate_basic():
    """
    Test that `tokenize_and_collate` correctly processes a small dataset by verifying:
    - Output dataset contains the required columns: ['input_ids', 'attention_mask', 'labels', 'weights']
    - Each sequence (`input_ids` and `attention_mask`) is padded/truncated to the specified max_length
    - Labels are stored as floats
    - Weights are stored as floats
    """

    data = {
        "text": ["Good product", "Bad quality"],
        "usefulness_score": [9.0, 2.0],
        "confidence_score": [0.9, 0.8],
    }
    dataset = Dataset.from_dict(data)
    tokenized_dataset = tokenize_and_collate(dataset, tokenizer, max_length=16)
    
    for col in ['input_ids', 'attention_mask', 'labels', 'weights']:
        assert col in tokenized_dataset.column_names
    
    for row in tokenized_dataset:
        assert row['input_ids'].shape[0] == 16
        assert row['attention_mask'].shape[0] == 16
        assert isinstance(row['labels'], float)
        assert isinstance(row['weights'], float)

def test_tokenize_and_collate_missing_weights():
    """
    Test that `tokenize_and_collate` correctly processes a small dataset with missing weights column by verifying:
    - Weights are stored as 1.0 for all rows
    """
    data = {
        "text": ["Nice!", "Terrible!"],
        "usefulness_score": [8.0, 1.0],
    }
    dataset = Dataset.from_dict(data)
    tokenized_dataset = tokenize_and_collate(dataset, tokenizer, max_length=16)
    for row in tokenized_dataset:
        assert row['weights'] == 1.0

def test_tokenize_and_collate_partial_weights():
    """
    Test that `tokenize_and_collate` correctly processes a small dataset with some rows missing weights value by verifying:
    - Weights are stored as 1.0 for all missing rows
    """
    data = {
        "text": ["Good", "Bad"],
        "usefulness_score": [7.0, 3.0],
        "confidence_score": [None, 0.5]
    }
    dataset = Dataset.from_dict(data)
    tokenized_dataset = tokenize_and_collate(dataset, tokenizer, max_length=16)
    weights = [row['weights'] for row in tokenized_dataset]
    assert weights[0] == 1.0
    assert weights[1] == 0.5



def test_weighted_mse_loss_equal_weights():
    """Test the working of loss function(weighted MSE)"""
    loss_fn = WeightedMSELoss()
    labels = torch.tensor([1.0, 2.0, 3.0])
    outputs = torch.tensor([1.1, 1.9, 3.2])
    weights = torch.ones_like(labels)

    weighted_loss = loss_fn(labels, outputs, weights)
    
    mse_loss = nn.MSELoss()(outputs, labels)
    assert torch.isclose(weighted_loss, mse_loss, atol=1e-6)

def test_weighted_mse_loss_larger_weights():
    """Test the loss function when weights are not 1."""
    loss_fn = WeightedMSELoss()
    labels = torch.tensor([1.0, 2.0, 3.0])
    outputs = torch.tensor([1.1, 1.9, 3.2])
    weights = torch.tensor([2.0, 2.0, 2.0])
    
    weighted_loss = loss_fn(labels, outputs, weights)
    mse_loss = nn.MSELoss()(outputs, labels)
    assert torch.isclose(weighted_loss, 2*mse_loss, atol=1e-6)

def test_weighted_mse_loss_no_weights():
    loss_fn = WeightedMSELoss()
    labels = torch.tensor([1.0, 2.0, 3.0])
    outputs = torch.tensor([1.1, 1.9, 3.2])
    
    loss = loss_fn(labels, outputs) 
    mse_loss = nn.MSELoss()(outputs, labels)
    assert torch.isclose(loss, mse_loss, atol=1e-6)

def test_weighted_mse_loss_single_sample():
    loss_fn = WeightedMSELoss()
    labels = torch.tensor([2.5])
    outputs = torch.tensor([3.0])
    weights = torch.tensor([0.5])
    
    loss = loss_fn(labels, outputs, weights)
    expected = ((3.0-2.5)**2 * 0.5)
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

def test_weighted_mse_loss_float_weights():
    loss_fn = WeightedMSELoss()
    labels = torch.tensor([1.0, 2.0, 3.0])
    outputs = torch.tensor([0.9, 2.1, 2.8])
    weights = torch.tensor([0.3, 0.7, 1.2])
    
    loss = loss_fn(labels, outputs, weights)
    expected = (((0.9-1.0)**2*0.3 + (2.1-2.0)**2*0.7 + (2.8-3.0)**2*1.2)/3)
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)

def test_weighted_mse_loss_backward():
    loss_fn = WeightedMSELoss()
    labels = torch.tensor([1.0, 2.0, 3.0], requires_grad=False)
    outputs = torch.tensor([1.1, 1.9, 3.2], requires_grad=True)
    weights = torch.tensor([1.0, 0.5, 2.0])
    
    loss = loss_fn(labels, outputs, weights)
    loss.backward()
    assert outputs.grad is not None
    assert outputs.grad.shape == outputs.shape



class DummyModel(nn.Module):
    def __init__(self, outputs):
        super().__init__()
        self._outputs = outputs

    def __call__(self, **kwargs):
        # Mimic HF Trainer return style: Namespace or dict with logits
        return SimpleNamespace(logits=self._outputs)

def test_compute_loss_scalar_output():
    """Builds a fake batch with input_ids, attention_mask, labels, and weights.
        Defines a DummyModel that ignores inputs and just returns some fake logits ([1.5, 2.5]).
        Calls your compute_loss function with that batch + model + loss function.
        Checks that the output is a scalar tensor (i.e. a single number, not a vector or matrix).
    """
    batch = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.tensor([[1, 1], [1, 1]]),
        "labels": torch.tensor([1.0, 2.0]),
        "weights": torch.tensor([1.0, 1.0]),
    }

    model = DummyModel(outputs=torch.tensor([1.5, 2.5]))
    loss_fn = WeightedMSELoss()
    loss = compute_loss(model, batch, loss_fn)

    assert torch.is_tensor(loss)
    assert loss.dim() == 0  # scalar

def test_compute_loss_decreases_when_predictions_match_labels():
    """Makes the same fake batch (labels = [1.0, 2.0]).
        Defines two models:
        bad_model → outputs [10.0, -5.0] (very far from labels).
        good_model → outputs [1.0, 2.0] (exactly matches labels).
        Computes the loss for both.
        Asserts that the good model’s loss is smaller than the bad model’s loss.
    """
    batch = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "attention_mask": torch.tensor([[1, 1], [1, 1]]),
        "labels": torch.tensor([1.0, 2.0]),
        "weights": torch.tensor([1.0, 1.0]),
    }

    bad_model = DummyModel(outputs=torch.tensor([10.0, 9.0]))
    good_model = DummyModel(outputs=torch.tensor([1.5, 0.0]))

    loss_fn = WeightedMSELoss()

    bad_loss = compute_loss(bad_model, batch, loss_fn)
    good_loss = compute_loss(good_model, batch, loss_fn)

    assert good_loss < bad_loss, f"Expected good loss {good_loss} < bad loss {bad_loss}"

