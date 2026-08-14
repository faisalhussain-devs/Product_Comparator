# Product Comparator

**An advanced NLP system for structured product opinion mining, review evidence ranking, hierarchical aspect-based sentiment analysis, and product comparison.**

Product Comparator takes noisy user-generated product reviews and transforms them into structured, aspect-level evidence that can be combined with concrete product specifications.

The project was built as an **NLP/ML engineering system**, not as a simple sentiment-analysis application. The core work is in the Transformer architecture, hierarchical label modeling, training strategy, class-imbalance handling, threshold optimization, evidence ranking, confidence estimation, and optimized inference.

> **Current version:** focused on the NLP/ML pipeline. The web interface is intentionally out of scope.

---

## Overview

A conventional sentiment classifier might turn:

> "The camera is excellent, but battery life is disappointing."

into:

```text
negative
```

That is not sufficient for product comparison.

Product Comparator tries to preserve the structure of the opinion:

```text
Camera
└── Camera Quality
    └── Positive

Battery & Charging
└── Battery Life & Health
    └── Negative
```

The system therefore treats a review as a collection of **evidence about product attributes**, rather than as a single positive/negative document.

---

# End-to-End Architecture

```text
                         ┌──────────────────┐
                         │   Product Query  │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │ Product / Data Resolution│
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Review Preprocessing     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Product Relevance Filter │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Transformer Review       │
                    │ Usefulness Ranker         │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Evidence Selection       │
                    └────────────┬─────────────┘
                                 │
                                 ▼
          ┌────────────────────────────────────────────┐
          │        Hierarchical Transformer ABSA       │
          │                                            │
          │  Top-level aspect → Sub-aspect → Sentiment │
          │                                            │
          │  Aspect-specific attention + task heads    │
          └──────────────────────┬─────────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Confidence / Evidence    │
                    │ Aggregation               │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Specification Integration│
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │ Structured Product       │
                    │ Analysis                 │
                    └──────────────────────────┘
```

---

# What makes the NLP pipeline different

The system is composed of multiple learned and engineered stages rather than one black-box classifier.

### Review-level reasoning

```text
Is this review about the requested product?
                ↓
Is it useful evidence?
                ↓
What aspect does it discuss?
                ↓
Which sub-aspect?
                ↓
What sentiment is expressed?
                ↓
How confident is the evidence?
```

This separation is intentional.

A review can be highly relevant to a product while providing very little useful evidence. Conversely, a short review can contain extremely valuable information about a specific aspect.

---

# 1. Transformer Review Usefulness Ranking

Before running expensive aspect analysis, candidate reviews are ranked using a trained Transformer-based usefulness model.

```text
Candidate Reviews
       │
       ▼
Product Relevance
       │
       ▼
Transformer Usefulness Score
       │
       ▼
Ranked Evidence
       │
       ▼
Hierarchical ABSA
```

This prevents the downstream NLP system from treating hundreds of reviews as equally informative.

The production implementation performs batched model inference and selects high-value evidence for downstream analysis.

The ranking model is therefore a distinct learned component of the system, rather than an arbitrary heuristic such as sorting by upvotes.

---

# 2. Hierarchical Aspect-Based Sentiment Analysis

The central NLP component is a **hierarchical multi-task Transformer architecture**.

Instead of predicting a flat set of unrelated labels, the model learns a product-specific hierarchy:

```text
Top-level Aspect
       │
       ▼
   Sub-aspect
       │
       ▼
    Sentiment
```

The taxonomy contains a set of broad product categories and their corresponding fine-grained sub-aspects.

Representative categories include:

```text
Battery & Charging
├── battery capacity
├── battery charging speed
└── battery life & health

Camera
├── camera features
└── camera quality

Device Performance
├── processor performance
├── technical performance & storage specs
└── thermals

Display
├── display aesthetics
├── display defects
├── display quality
└── display visual performance
```

The final model selected for the system was chosen after multiple training and optimization experiments rather than simply using the last checkpoint produced by training.

---

# 3. Shared Transformer Encoder

The model begins with a domain-trained Transformer encoder.

Rather than building independent models for every aspect, the architecture shares linguistic representations across the task while adding task-specific components above the encoder.

Conceptually:

```text
                    Review Text
                        │
                        ▼
               Tokenization / Encoder
                        │
                        ▼
              Transformer Representations
                        │
             ┌──────────┼──────────┐
             │          │          │
             ▼          ▼          ▼
        Top Aspect   Sub Aspect  Sentiment
          Head         Head        Head
```

The shared encoder allows different product aspects to benefit from common language representations while the specialized heads learn the distinctions required by each task.

---

# 4. Aspect-Specific Attention

A central part of the architecture is **aspect-conditioned attention**.

Instead of relying only on a pooled representation of the entire review, the model learns representations conditioned on the aspect being considered.

```text
                 Review Tokens
                      │
                      ▼
             Transformer Encoder
                      │
          ┌───────────┴───────────┐
          │                       │
     Aspect Query            Token States
          │                       │
          └───────────┬───────────┘
                      ▼
               Attention
                      │
                      ▼
            Aspect-specific Context
                      │
                      ▼
                Task Head
```

This is important for reviews containing multiple opinions:

> "The display is excellent, the speakers are average, and battery life is poor."

The model needs to attend to different parts of the review depending on the aspect being analyzed.

---

# 5. Hierarchical Top → Sub-Aspect Modeling

The sub-aspect classifier does not operate as an entirely independent flat 29-class problem.

The system maintains an explicit relationship between higher-level aspects and their valid sub-aspects.

```text
Top-level prediction
        │
        ▼
Top → Sub mapping
        │
        ▼
Relevant sub-aspect space
        │
        ▼
Fine-grained prediction
```

This introduces structural information into the classifier.

For example:

```text
Camera
├── Camera Quality
└── Camera Features
```

is fundamentally different from treating:

```text
Camera Quality
Battery Capacity
Thermals
Charging Speed
Display Quality
...
```

as 29 unrelated classes.

The hierarchy is used as part of the training/inference design.

---

# 6. Handling Class Imbalance

The sub-aspect dataset is highly imbalanced.

Some aspects appear thousands of times while others have substantially fewer examples.

The training system addresses this at multiple levels.

### Weighted sampling

A `WeightedRandomSampler` is used to increase exposure to examples containing underrepresented top-level aspects.

Conceptually:

```text
Frequent aspect
      ↓
Lower sampling probability

Rare aspect
      ↓
Higher sampling probability
```

The sampling weight is derived from inverse frequency of the active top-level labels.

### Weighted focal loss

The sub-aspect task also uses a weighted focal-loss formulation.

This allows the training objective to focus more heavily on difficult and underrepresented examples rather than optimizing primarily for dominant classes.

This combination means imbalance is addressed through both:

```text
data sampling
      +
loss weighting
```

rather than relying on one technique.

---

# 7. Staged Encoder Unfreezing

The Transformer encoder is not immediately allowed to update at full capacity.

Training begins with task-specific components being optimized while the encoder is constrained, followed by full encoder unfreezing.

Observed training schedule:

```text
Epoch 1
   ↓
Epoch 2
   ↓
Unfreeze full encoder
   ↓
Epoch 3+
Full Transformer fine-tuning
```

This separates initial task-head stabilization from deeper domain adaptation.

It also has a real computational consequence: once the full encoder is unfrozen, training becomes substantially more expensive.

---

# 8. Layer-Wise Learning-Rate Decay

The Transformer does not necessarily need every layer to move at the same rate.

The training strategy uses **layer-wise learning-rate decay**, allowing different parts of the pretrained encoder to adapt at different rates.

The principle is:

```text
Task heads / upper layers
        ↓
larger adaptation

Lower Transformer layers
        ↓
smaller adaptation
```

This is intended to preserve useful general linguistic representations in lower layers while allowing higher-level representations to adapt more strongly to the product-domain task.

---

# 9. Targeted Logit-Margin Separation

An additional experiment explored how to make semantically similar sibling classes more separable.

Instead of only requiring:

```text
logit(true) > logit(false)
```

the margin formulation encourages:

```text
logit(true_sub) ≥ logit(false_sub) + m
```

This is particularly useful for confusable sibling aspects.

For example:

```text
Camera
├── Camera Quality
└── Camera Features
```

or:

```text
Battery & Charging
├── Battery Capacity
├── Charging Speed
└── Battery Life & Health
```

The goal is to improve separation between sibling predictions without forcing the learned representations themselves to become artificially orthogonal.

This was chosen after experimentation with alternative representation-separation ideas.

---

# 10. Sentiment Modeling

Sentiment is predicted at the aspect/sub-aspect level.

The system therefore produces structured outputs such as:

```text
Camera Quality
    → Positive

Battery Life & Health
    → Negative

Build Quality
    → Positive

Thermals
    → Negative
```

rather than assigning one sentiment label to the entire review.

The sentiment component uses aspect-conditioned representations and a task-specific head.

The training objective includes both classification loss and targeted margin separation.

---

# 11. Per-Class Threshold Optimization

One of the major evaluation improvements was moving away from a universal classification threshold.

A default classifier might use:

```text
P(class) >= 0.5
```

for every label.

That assumption does not hold well for the project's imbalanced multi-label setting.

The final evaluation therefore performs **5-fold threshold optimization** and obtains a separate decision threshold for each class.

Example:

```text
Class 0 → 0.788
Class 1 → 0.794
Class 2 → 0.434
Class 3 → 0.256
Class 4 → 0.888
...
Class 9 → 0.540
```

This produces a much more appropriate operating point for classes with different score distributions and prevalence.

### Best threshold-optimized result

```text
Macro F1: 0.7929
```

with per-class precision, recall and F1 reported individually.

Representative results:

| Class | Threshold | F1 | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0 | 0.788 | 0.8765 | 0.8659 | 0.8875 |
| 1 | 0.794 | 0.7948 | 0.8053 | 0.7845 |
| 2 | 0.434 | 0.8682 | 0.8058 | 0.9412 |
| 3 | 0.256 | 0.8145 | 0.7434 | 0.9006 |
| 4 | 0.888 | 0.5631 | 0.5577 | 0.5686 |
| 5 | 0.526 | 0.7707 | 0.7143 | 0.8368 |
| 6 | 0.606 | 0.7773 | 0.7773 | 0.7773 |
| 7 | 0.724 | 0.8220 | 0.7697 | 0.8819 |
| 8 | 0.718 | 0.8120 | 0.7883 | 0.8372 |
| 9 | 0.540 | 0.8297 | 0.7901 | 0.8734 |

The selected production model was chosen from the broader set of training experiments based on the best observed validation performance.

---

# Training Results

The development process was iterative rather than a single training run.

The progression of the selected hierarchical training experiment shows the effect of the training strategy:

| Epoch | Sub-aspect Macro F1 | Hierarchy Accuracy | Top-level F1 |
|---:|---:|---:|---:|
| 1 | 0.4117 | 0.7629 | 0.7970 |
| 2 | 0.5417 | 0.8285 | 0.7970 |
| 3 | 0.6129 | 0.8499 | 0.7916 |
| 4 | 0.6513 | 0.8777 | 0.7785 |
| **5** | **0.6845** | **0.8878** | **0.7910** |

The important observation is that the hierarchy becomes substantially more accurate as the fine-grained classifier learns:

```text
Sub-aspect Macro F1
0.4117 → 0.6845

Hierarchy Accuracy
0.7629 → 0.8878
```

The final production model is selected from the broader experiment set using the best validation result, rather than assuming that the final training epoch is automatically the best model.

---

# Training Stack

The training pipeline combines:

- domain-trained Transformer encoder
- task-specific classification heads
- aspect-specific attention
- hierarchical top/sub supervision
- weighted multi-label learning
- inverse-frequency sampling
- weighted focal loss
- AdamW
- layer-wise learning-rate decay
- differentiated learning rates
- staged encoder unfreezing
- mixed-precision training
- gradient clipping
- learning-rate warmup and decay
- targeted logit-margin separation
- per-class threshold optimization
- checkpoint selection based on validation performance

This combination is the main ML engineering component of the project.

---

# Confidence and Evidence Aggregation

The final system does not simply expose raw classifier logits.

Predicted aspects are connected back to the reviews that generated the evidence.

Conceptually:

```text
Review
   │
   ├── usefulness score
   │
   ├── aspect prediction
   │
   └── sentiment prediction
          │
          ▼
       Evidence
          │
          ▼
   Confidence aggregation
          │
          ▼
   Product-level conclusion
```

This allows the system to distinguish between an aspect supported by many useful reviews and an aspect supported by sparse evidence.

---

# Specification Fusion

Reviews provide subjective evidence.

Specifications provide objective product attributes.

The pipeline combines both:

```text
                Product
                   │
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
User Reviews             Specifications
        │                     │
        ▼                     ▼
NLP-derived opinions     Structured facts
        │                     │
        └──────────┬──────────┘
                   ▼
           Product Analysis
```

This makes the final representation useful for comparison rather than limiting it to sentiment analysis.

---

# Optimized Inference

Training and inference are deliberately separated.

```text
              Training
                 │
                 ▼
        PyTorch Transformer
                 │
                 ▼
               ONNX
                 │
                 ▼
          ONNX Runtime
             │       │
             ▼       ▼
           CUDA     CPU
```

The runtime uses ONNX Runtime for production inference, including graph optimization and selectable CUDA/CPU execution providers.

Large model artifacts are hosted separately from the source repository.

This keeps Git focused on source code, training methodology, configuration and reproducible project structure rather than hundreds of megabytes of model binaries.

---

# End-to-End Runtime

For a query such as:

```bash
python -m src.product_comparator.data_collection "iPhone 15 Pro"
```

the pipeline performs:

```text
1. Product resolution
2. Corpus/cache lookup
3. Review preprocessing
4. Product relevance filtering
5. Transformer review ranking
6. Evidence selection
7. Hierarchical aspect prediction
8. Sub-aspect prediction
9. Aspect-level sentiment
10. Confidence calculation
11. Specification integration
12. Structured result generation
```

The output contains product information, ranked reviews, aspect-level analysis, supporting evidence, confidence information and specifications.

---

# Engineering Decisions

Several design decisions were made specifically to keep the system usable beyond an offline notebook.

### Separate training and inference

Training uses the full PyTorch stack and experimentation infrastructure.

Inference uses optimized ONNX models.

### Batch model execution

Reviews are processed in batches rather than one request at a time.

### Model caching

Large model artifacts are not repeatedly downloaded during normal operation.

### Packaged product corpus

Common products can be resolved locally without requiring live data collection.

### Hierarchical label representation

The taxonomy is represented explicitly rather than encoded only as unrelated class IDs.

### Per-class thresholds

Decision thresholds are learned from validation behavior rather than hard-coded universally.

### Evidence-aware output

Predictions retain links to the reviews that support them.

---

# Project Structure

```text
Product_Comparator/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── src/
│   └── product_comparator/
│       ├── data_collection.py
│       ├── dataset_cache.py
│       ├── preprocessing.py
│       ├── relevance.py
│       ├── ranking.py
│       ├── habsa.py
│       ├── confidence.py
│       ├── inference.py
│       ├── models.py
│       ├── reddit_collector.py
│       ├── gsmarena_collector.py
│       └── config.py
│
├── experiments/
│   ├── training/
│   ├── confidence/
│   ├── threshold_optimization/
│   └── model_export_onnx/
│
├── example_data/
│   ├── build_dataset_corpus.py
│   ├── fetch_reddit_product_data.py
│   ├── product_index.json
│   └── products_corpus.json
│
└── tests/
```

---

# Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the end-to-end pipeline:

```bash
python -m src.product_comparator.data_collection "iPhone 15 Pro"
```

Training experiments are available under:

```text
experiments/training/
```

The production configuration is located at:

```text
src/product_comparator/config.py
```

---

# Model Artifacts

The trained production models and tokenizer artifacts are hosted separately.

The runtime is configured to load the trained artifacts from the associated Hugging Face repository rather than storing large model binaries inside Git.

This repository therefore contains the **architecture, training code, inference pipeline, evaluation methodology and project logic**, while large checkpoints remain external.

---

# Current Limitations

The current implementation is primarily an NLP/ML research and engineering system rather than a finished consumer product.

One known limitation is the product relevance stage. Closely related product generations and variants can still create false-positive review matches. This is a known area for future refinement.

The web interface is intentionally out of scope for the current version.

---

# What this project demonstrates

The project covers the complete lifecycle of a non-trivial NLP system:

```text
Problem formulation
        ↓
Domain data construction
        ↓
Hierarchical label design
        ↓
Transformer architecture
        ↓
Attention design
        ↓
Multi-task training
        ↓
Class imbalance handling
        ↓
Training stabilization
        ↓
Hyperparameter / threshold optimization
        ↓
Validation and model selection
        ↓
ONNX conversion
        ↓
Optimized inference
        ↓
Evidence aggregation
        ↓
Structured product intelligence
```

The emphasis is not on using a pretrained Transformer as a black box.

The main engineering work is in **designing how the Transformer is trained, how hierarchical product concepts are represented, how difficult and imbalanced labels are handled, how predictions are calibrated, and how the resulting models are moved into a practical inference pipeline.**

---

# License

This project is released under the MIT License. See `LICENSE`.
