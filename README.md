# Product Comparator

### Transformer-based Product Opinion Mining, Hierarchical ABSA & Evidence Ranking

Product Comparator is an **advanced NLP/ML system for turning noisy, user-generated product discussions into structured product intelligence**.

Instead of assigning a single sentiment to an entire review, the system learns to identify **what product aspect is being discussed, which fine-grained sub-aspect is involved, and what sentiment is expressed**, while separately ranking reviews by their usefulness as evidence.

The project combines:

- a domain-trained Transformer encoder;
- aspect-conditioned attention;
- hierarchical top-level → sub-aspect modeling;
- multi-label classification;
- hierarchy-aware masking;
- weighted focal loss;
- inverse-frequency sampling;
- sibling-aware hard-negative margin loss;
- staged Transformer unfreezing;
- layer-wise learning-rate decay;
- task-specific adapters;
- per-class threshold optimization;
- confidence/evidence aggregation;
- ONNX export and optimized inference.

> **Project focus:** NLP/ML engineering and model development. The web interface is intentionally out of scope for this version.

---

# 1. Problem

Product reviews are not naturally organized data.

A Reddit discussion can contain:

- multiple products in the same thread;
- incomplete sentences;
- slang and informal language;
- contradictory opinions;
- short comments with little context;
- long reviews covering many aspects;
- product-specific terminology;
- irrelevant discussion;
- subjective comparisons;
- nested replies;
- noisy punctuation, URLs and symbols.

A useful product-analysis system therefore needs to answer several different questions.

```text
                    Review
                       │
                       ▼
             Is it about this product?
                       │
                       ▼
             Is it useful evidence?
                       │
                       ▼
             What aspect is discussed?
                       │
                       ▼
             What sub-aspect is discussed?
                       │
                       ▼
             What sentiment is expressed?
                       │
                       ▼
             How strong is the evidence?
```

Product Comparator was designed around this decomposition.

---

# 2. The Core Idea

A conventional sentiment model might turn:

> "The camera is amazing but the battery is disappointing."

into:

```text
negative
```

That is almost useless for a product comparison engine.

Product Comparator attempts to preserve the structure:

```text
Camera
└── Camera Quality
    └── Positive

Battery & Charging
└── Battery Life & Health
    └── Negative
```

The output is therefore not simply:

```text
positive / negative
```

but structured evidence that can be aggregated across many reviews.

---

# 3. End-to-End Architecture

```text
                         Product Query
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Product / Data Resolution│
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Review Preprocessing    │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Product Relevance       │
                 │ Filtering               │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Transformer Review      │
                 │ Usefulness Ranker       │
                 └────────────┬────────────┘
                              │
                              ▼
                     Evidence Selection
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │      Hierarchical Transformer NLP Model   │
        │                                           │
        │   Top-level Aspect                        │
        │          │                                │
        │          ▼                                │
        │      Sub-aspect                           │
        │          │                                │
        │          ▼                                │
        │       Sentiment                           │
        │                                           │
        │  Aspect-conditioned attention              │
        │  Hierarchy-aware constraints               │
        │  Task-specific heads/adapters             │
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
                  Confidence / Evidence
                     Aggregation
                              │
                              ▼
                  Specification Fusion
                              │
                              ▼
                  Structured Product
                       Analysis
```

---

# 4. Data: Real, Noisy User-Generated Language

A major part of the project was deliberately not built around a clean benchmark corpus.

The system was trained/evaluated using **real-world product discussion data collected from Reddit**, together with product information and specifications.

This matters because Reddit data is inherently noisy.

The preprocessing pipeline explicitly handles:

- URLs;
- emojis and symbols;
- inconsistent punctuation;
- escaped/newline characters;
- repeated separators;
- irregular whitespace;
- comment fragments;
- review/comment text with inconsistent formatting.

The repository contains dedicated Reddit collection and preprocessing components, and the preprocessing code converts raw review/comment structures into cleaned text suitable for downstream modeling. fileciteturn25file0L1-L5 fileciteturn25file1L6-L10 fileciteturn26file0L2-L6

This makes the modeling problem materially different from training on perfectly curated sentences.

### Data flow

```text
Reddit discussions
       │
       ▼
Product-linked reviews/comments
       │
       ▼
Cleaning + normalization
       │
       ▼
Sentence/comment preprocessing
       │
       ▼
Aspect / sub-aspect / sentiment labels
       │
       ▼
Transformer training
```

The model is therefore intended to learn product-domain semantics from **messy, naturally occurring language**, rather than only from clean textbook-style examples.

---

# 5. Hierarchical Product Taxonomy

The NLP problem is represented hierarchically.

```text
Top-level aspect
       │
       ├───────────────┐
       ▼               ▼
Sub-aspect A      Sub-aspect B
       │               │
       └───────┬───────┘
               ▼
           Sentiment
```

Representative product categories include:

```text
Battery & Charging
├── Battery Capacity
├── Charging Speed
└── Battery Life & Health

Camera
├── Camera Features
└── Camera Quality

Device Performance
├── Processor Performance
├── Technical Performance & Storage Specs
└── Thermals

Display
├── Display Aesthetics
├── Display Defects
├── Display Quality
└── Display Visual Performance
```

The hierarchy is not merely documentation.

It is represented in the model and inference pipeline through:

- separate top-level and sub-aspect heads;
- learned aspect query vectors;
- top → sub relationships;
- hierarchy-aware sub-aspect masking;
- sibling-aware margin learning.

---

# 6. Transformer Architecture

## Shared Domain Encoder

The model uses a domain-trained Transformer encoder as the shared representation backbone.

The general architecture is:

```text
                    Review Text
                        │
                        ▼
                 Tokenizer
                        │
                        ▼
              Domain Transformer
                        │
                Token representations
                        │
           ┌────────────┼────────────┐
           │            │            │
           ▼            ▼            ▼
      Top Aspect    Sub-aspect   Sentiment
         Head          Head         Head
```

The encoder provides shared language representations while the task-specific components specialize those representations for product reasoning.

---

# 7. Aspect-Conditioned Attention

One of the main architectural components is a custom `AspectAttention` module.

Rather than relying only on the pooled Transformer representation, the model maintains learned aspect/query vectors and uses them to attend over the token-level encoder representation.

Conceptually:

```text
                 Aspect Query
                      │
                      ▼
                    Wq
                      │
                      ▼
                 Query Vector
                      │
                      │
Review Tokens ──► Transformer ──► Token States
                                      │
                               ┌──────┴──────┐
                               │             │
                              Wk            Wv
                               │             │
                               ▼             ▼
                         Attention Scores  Values
                               │             │
                               └──────┬──────┘
                                      ▼
                             Aspect Context
                                      │
                                      ▼
                               Task-specific Head
```

The implementation explicitly computes query/key/value projections and attends from aspect representations to the sequence representation. fileciteturn18file0L2-L2

This is particularly useful for reviews containing multiple opinions:

> "The display is fantastic, the speakers are average, and battery life is terrible."

Different aspect queries can focus on different portions of the same review.

---

# 8. Learned Aspect Query Representations

The architecture maintains learned query representations for the product aspects.

Instead of asking the same pooled representation to solve every classification problem, the model creates an aspect-conditioned representation for each candidate label.

This makes the classifier more structured:

```text
                    Shared Encoder
                          │
          ┌───────────────┼────────────────┐
          │               │                │
          ▼               ▼                ▼
       Camera          Battery          Display
       Query             Query            Query
          │               │                │
          ▼               ▼                ▼
    Camera context   Battery context   Display context
```

The training code also includes experiments for measuring the cosine similarity of learned query vectors and checking whether different label representations collapse toward one another. fileciteturn18file0L2-L2

---

# 9. Hierarchy-Aware Sub-Aspect Prediction

The sub-aspect task is not treated as a completely independent flat classification problem.

The model knows that certain sub-aspects belong to certain top-level aspects.

For example:

```text
Camera
├── Camera Quality
└── Camera Features
```

means that an unrelated sub-aspect such as:

```text
Battery Capacity
```

should not become a candidate simply because its classifier score happens to be high.

The inference pipeline therefore applies a top → sub mask:

```text
Top-level probabilities
          │
          ▼
Class thresholds
          │
          ▼
Predicted top aspects
          │
          ▼
Top → sub mapping
          │
          ▼
Allowed sub-aspects
          │
          ▼
Masked sub-aspect probabilities
```

The implementation explicitly constructs the mask from predicted top-level labels and the top-to-sub mapping. fileciteturn21file0L2-L2

If no top-level aspect is predicted, the implementation retains a fallback that allows all sub-aspects rather than forcing an empty prediction.

---

# 10. Multi-Label Learning

Product reviews can discuss multiple aspects simultaneously.

For example:

```text
Camera       = active
Battery      = active
Display      = inactive
Thermals     = active
```

The model therefore uses multi-label outputs rather than forcing each review into exactly one aspect.

The training data represents top-level labels as multi-hot vectors and sub-aspects as sparse binary structures. fileciteturn21file0L2-L2

This is an important distinction from ordinary single-label text classification.

---

# 11. Class Imbalance

The product taxonomy is not uniformly represented.

Some aspects appear very frequently in Reddit discussions, while others are rare.

The training system attacks this problem at multiple levels.

## 11.1 Weighted Sampling

The training pipeline uses inverse-frequency weighting to increase exposure to underrepresented examples.

Conceptually:

```text
Rare aspect
    ↓
higher sampling weight
    ↓
more training exposure

Common aspect
    ↓
lower relative weight
```

This prevents the optimizer from seeing the same dominant aspects disproportionately often.

---

## 11.2 Weighted Focal Loss

The sub-aspect and downstream tasks use a weighted focal-loss formulation.

The implementation combines:

```text
Binary cross-entropy
        +
Focal modulation
        +
Alpha balancing
        +
Class/sample weights
```

with:

```text
gamma = 1.5
alpha = 0.25
```

in the implemented focal-loss function. fileciteturn18file0L2-L2

The purpose is to reduce the influence of easy examples while concentrating optimization on difficult and underrepresented predictions.

---

# 12. Hierarchy-Aware Sibling Margin Loss

One of the more specialized parts of the final training design is the sub-aspect margin objective.

Rather than enforcing a margin between arbitrary labels, the implementation restricts the comparison to **sibling sub-aspects under the same top-level aspect**.

The objective is conceptually:

```text
positive sibling logit
        ≥
hard-negative sibling logit
        +
margin
```

The implementation:

1. activates only samples belonging to a particular top-level aspect;
2. selects that top's valid sub-aspects;
3. identifies positive and negative sibling labels;
4. selects hard negative logits;
5. applies a margin violation penalty.

The implemented default margin is `0.3`, with hard-negative selection over the strongest competing siblings. fileciteturn18file0L2-L2

This is more targeted than simply trying to make every class representation orthogonal.

### Why sibling-only?

Consider:

```text
Camera
├── Camera Quality
└── Camera Features
```

The meaningful competition is:

```text
Camera Quality vs Camera Features
```

not:

```text
Camera Quality vs Battery Capacity
```

The loss therefore uses the taxonomy itself to define the difficult negative space.

---

# 13. Representation-Separation Experiments

The training experiments also explored explicit orthogonality regularization over learned label/query vectors.

The implementation computes a normalized Gram matrix and penalizes off-diagonal similarity. fileciteturn18file0L2-L2

However, this should be presented as an **experiment in the model-development process**, not as the headline claim of the final system.

The more useful final design is the hierarchy-aware sibling margin because it injects task structure into the separation objective instead of enforcing generic geometric separation everywhere.

This distinction is important:

```text
Generic representation separation
             vs.
Task-aware sibling separation
```

The latter is much easier to justify for a hierarchical product taxonomy.

---

# 14. Staged Fine-Tuning

The full Transformer is not blindly fine-tuned from the first optimization step.

The training process uses staged encoder unfreezing.

```text
Initial training
      │
      ▼
Task-specific components adapt
      │
      ▼
Training stabilizes
      │
      ▼
Full encoder unfreezing
      │
      ▼
Domain adaptation
```

The recorded training run explicitly unfreezes the full encoder at Epoch 3.

This allows newly initialized task-specific components to establish useful gradients before the entire pretrained representation is aggressively updated.

---

# 15. Layer-Wise Learning-Rate Decay

The Transformer layers do not all need to adapt equally.

The training strategy uses layer-wise learning-rate decay so that lower encoder layers receive smaller updates while higher layers and task-specific components can adapt more strongly.

Conceptually:

```text
Task heads
    │
    │ higher LR
    ▼
Upper Transformer layers
    │
    │
    ▼
Lower Transformer layers
    │
    │ lower LR
    ▼
General linguistic representations
```

This is particularly appropriate when adapting a pretrained language encoder to a relatively specialized product-review domain.

---

# 16. Task-Specific Sentiment Adapter

The sentiment stage introduces a lightweight residual adapter:

```text
Hidden representation
        │
        ├─────────────────────────┐
        │                         │
        ▼                         │
Down projection                  │
        │                         │
      GELU                       │
        │                         │
Up projection                    │
        │                         │
        └──────────┬──────────────┘
                   ▼
              Residual output
```

The implementation uses a bottleneck projection from the Transformer hidden dimension to a smaller representation and projects it back, with the final projection initially zero-initialized. fileciteturn21file0L2-L2

This gives the sentiment task a controlled adaptation path without requiring the sentiment head to completely redefine the shared representation.

---

# 17. Sentiment Modeling

Sentiment is predicted for detected product aspects rather than for the entire document.

Example:

```text
Review:
"The camera is great, battery life is poor,
and the display is excellent."

Output:

Camera Quality
    → Positive

Battery Life & Health
    → Negative

Display Quality
    → Positive
```

The sentiment stage also preserves the hierarchical relationship between top-level and sub-aspect predictions.

The implementation uses the top-level predictions and top-to-sub mapping to mask irrelevant sub-aspect probabilities before downstream reasoning. fileciteturn21file0L2-L2

---

# 18. Threshold Optimization

A universal threshold of `0.5` is a poor assumption for an imbalanced multi-label problem.

The project therefore performs **per-class threshold optimization**.

The optimization procedure uses cross-validation to find robust thresholds for individual classes.

Example from the selected evaluation:

```text
Class 0 → 0.788
Class 1 → 0.794
Class 2 → 0.434
Class 3 → 0.256
Class 4 → 0.888
Class 5 → 0.526
Class 6 → 0.606
Class 7 → 0.724
Class 8 → 0.718
Class 9 → 0.540
```

The resulting threshold-optimized evaluation achieved:

```text
Macro F1 = 0.7929
```

The large variation in optimal thresholds is itself evidence that a single global threshold would be inappropriate for this label space.

---

# 19. Model Selection

The project went through multiple experiments rather than treating the first working training run as the final model.

Training experiments explored:

- different loss formulations;
- class/sample weighting;
- focal-loss settings;
- representation-separation objectives;
- sibling margin separation;
- staged freezing/unfreezing;
- learning-rate strategies;
- hierarchical constraints;
- sentiment adaptation;
- threshold optimization;
- model export and inference optimization.

The final production model was selected from these experiments based on validation performance.

The important distinction is:

```text
Experiment
    ↓
Evaluate
    ↓
Modify training strategy
    ↓
Evaluate again
    ↓
Select best-performing model
```

rather than:

```text
Train once
    ↓
Publish checkpoint
```

---

# 20. Evaluation

A selected hierarchical training run showed the following progression:

| Epoch | Sub-aspect Macro F1 | Hierarchy Accuracy | Top-level F1 |
|---:|---:|---:|---:|
| 1 | 0.4117 | 0.7629 | 0.7970 |
| 2 | 0.5417 | 0.8285 | 0.7970 |
| 3 | 0.6129 | 0.8499 | 0.7916 |
| 4 | 0.6513 | 0.8777 | 0.7785 |
| **5** | **0.6845** | **0.8878** | **0.7910** |

The separate threshold-optimization evaluation reached:

```text
Macro F1 = 0.7929
```

with per-class precision, recall and F1.

### Selected threshold-optimized evaluation

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

The model-development process keeps the per-class metrics visible instead of hiding performance behind a single aggregate score.

---

# 21. Evidence Ranking

Aspect analysis is preceded by a separate review-usefulness stage.

The system therefore has two different notions of relevance:

```text
Product relevance
        ↓
"Is this actually about the product?"

Review usefulness
        ↓
"Is this useful evidence for analysis?"
```

This separation is important.

A review can mention an iPhone without providing meaningful information about its camera, battery, or performance.

The usefulness ranker therefore attempts to prioritize the evidence that should reach the expensive downstream NLP stages.

---

# 22. Confidence and Evidence Aggregation

The final system does not expose only raw model probabilities.

Predictions are connected back to the review evidence that produced them.

Conceptually:

```text
Useful Review
      │
      ├── Aspect
      ├── Sub-aspect
      ├── Sentiment
      └── Model confidence
             │
             ▼
       Evidence aggregation
             │
             ▼
       Product-level result
```

This allows the final output to distinguish between:

```text
Strong recurring evidence
```

and:

```text
Weak / sparse evidence
```

The repository also contains dedicated confidence-analysis and inference-certainty experiments, alongside model-export work. fileciteturn25file9L47-L50 fileciteturn25file10L51-L55

---

# 23. Product Specification Fusion

The system combines subjective review evidence with structured specifications.

```text
                 Product
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
   User Reviews            Specifications
        │                       │
        ▼                       ▼
   NLP evidence            Structured facts
        │                       │
        └───────────┬───────────┘
                    ▼
             Product analysis
```

This allows a comparison system to combine statements such as:

```text
"Battery lasts all day"
```

with concrete specification information such as battery capacity and charging characteristics.

---

# 24. Production Inference

Training and production inference are separated.

```text
                 Training
                    │
                    ▼
              PyTorch Model
                    │
                    ▼
                  ONNX
                    │
                    ▼
             ONNX Runtime
               │         │
               ▼         ▼
             CUDA       CPU
```

The repository includes dedicated ONNX export and quantization experiments. fileciteturn15file3L16-L19 fileciteturn15file4L21-L25

This keeps the production runtime independent from the full training environment.

---

# 25. Runtime Pipeline

Given:

```bash
python -m src.product_comparator.data_collection "iPhone 15 Pro"
```

the runtime performs approximately:

```text
1. Resolve product
2. Check packaged/indexed product data
3. Load or collect reviews
4. Clean and normalize review text
5. Apply product relevance filtering
6. Rank candidate reviews
7. Select useful evidence
8. Run hierarchical aspect analysis
9. Apply hierarchy constraints
10. Predict sub-aspects
11. Predict aspect-level sentiment
12. Calculate confidence
13. Attach supporting reviews
14. Integrate product specifications
15. Return structured product analysis
```

---

# 26. Engineering for Real-World Inference

The project does not stop at training accuracy.

The runtime also addresses deployment constraints:

### Batched inference

Reviews are processed in batches rather than invoking the model once for every review.

### ONNX Runtime

The trained models can be executed through ONNX Runtime instead of requiring the complete training stack.

### CUDA / CPU execution

The inference layer can select an appropriate execution provider.

### External model artifacts

Large checkpoints are hosted separately from the source repository.

### Packaged product corpus

Common products can be resolved without always requiring live collection.

### Product indexing

The packaged corpus includes a search index and alias-generation logic to resolve product names and variants efficiently.

---

# 27. Project Structure

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
│   │   ├── stage1_top_labels_train.ipynb
│   │   ├── stage2_sub_labels_train.ipynb
│   │   ├── stage3_sentiments_train.ipynb
│   │   └── stage4_comparisons_train.ipynb
│   │
│   ├── confidence/
│   ├── threshold_optimization/
│   └── model_export_onnx/
│
├── example_data/
│   ├── build_dataset_corpus.py
│   └── fetch_reddit_product_data.py
│
└── tests/
```

The repository keeps the training stages visible so the modeling decisions can be inspected rather than hidden behind an opaque final checkpoint. The four training notebooks are explicitly present in the repository's `experiments/training` area. fileciteturn13file0L1-L5 fileciteturn13file1L6-L10 fileciteturn13file2L11-L15

---

# 28. Reproducible Training Philosophy

The experiments directory is intentionally part of the project.

It documents the progression from:

```text
baseline
   ↓
loss experiments
   ↓
imbalance handling
   ↓
hierarchical constraints
   ↓
representation experiments
   ↓
margin separation
   ↓
fine-tuning strategy
   ↓
threshold optimization
   ↓
model selection
   ↓
deployment
```

This is important because the main engineering contribution is not simply the final architecture.

It is the process of discovering which combination of:

- architecture;
- objective;
- sampling;
- optimization;
- hierarchy;
- calibration; and
- inference strategy

actually works on noisy product-review data.

---

# 29. What I Learned From the Model Development

The project exposed several practical problems that are easy to miss when working only with clean NLP benchmarks.

### A single threshold is not enough

Different labels develop very different probability distributions.

### Class imbalance affects more than loss

Sampling strategy and decision thresholds matter as well.

### Hierarchies are useful inductive bias

The top-level taxonomy can reduce the effective search space for fine-grained predictions.

### Generic representation separation is not necessarily the right objective

Making every label vector orthogonal is less targeted than separating genuinely confusable siblings.

### Fine-tuning the entire encoder immediately is not always desirable

Staged unfreezing provides a more controlled adaptation path.

### Real-world text requires engineering around the model

Preprocessing, evidence selection, caching, indexing and inference optimization matter as much as the classifier itself.

---

# 30. Limitations

This is an NLP/ML engineering project, not a claim of a solved general-purpose product intelligence problem.

Known limitations include:

- Reddit discussions can contain ambiguous product references;
- closely related product generations can cause relevance errors;
- noisy user-generated language can contain insufficient context;
- rare sub-aspects remain harder to classify;
- confidence is an evidence-oriented estimate, not a formal probability of correctness;
- live data sources can change independently of the model;
- the current version does not include the planned consumer-facing web interface.

These limitations are intentionally visible because the goal of the project is to demonstrate the underlying NLP system rather than hide its failure modes behind a UI.

---

# 31. What This Project Demonstrates

The project covers a substantial part of the lifecycle of a practical NLP system:

```text
Real-world noisy data
        ↓
Data cleaning
        ↓
Domain taxonomy design
        ↓
Hierarchical label representation
        ↓
Transformer architecture
        ↓
Custom attention
        ↓
Multi-label learning
        ↓
Class-imbalance mitigation
        ↓
Custom loss engineering
        ↓
Staged fine-tuning
        ↓
Hyperparameter / threshold experiments
        ↓
Validation-driven model selection
        ↓
Confidence analysis
        ↓
ONNX conversion
        ↓
Optimized inference
        ↓
Evidence aggregation
        ↓
Structured product intelligence
```

The important part is that the Transformer is **not being used as a black-box sentiment classifier**.

The engineering work is in designing the task structure around it.

---

# 32. Summary

Product Comparator combines:

```text
Domain-trained Transformer
          +
Aspect-conditioned attention
          +
Hierarchical multi-label modeling
          +
Top → sub constraints
          +
Weighted focal learning
          +
Inverse-frequency sampling
          +
Sibling-aware hard-negative margin
          +
Staged encoder fine-tuning
          +
Layer-wise LR decay
          +
Task-specific sentiment adapter
          +
Per-class threshold optimization
          +
Evidence ranking
          +
Confidence aggregation
          +
ONNX / optimized inference
```

The result is a product-analysis NLP pipeline designed to extract **structured, evidence-backed opinions from noisy real-world product discussions**.

---

# Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the current inference pipeline:

```bash
python -m src.product_comparator.data_collection "iPhone 15 Pro"
```

Training experiments are located under:

```text
experiments/training/
```

Runtime configuration is located under:

```text
src/product_comparator/config.py
```

---

# Model Artifacts

Large trained model artifacts are hosted separately from the source repository.

The runtime loads the configured Transformer/tokenizer artifacts and trained checkpoints from the associated model repository.

Keeping model binaries outside Git keeps the repository focused on:

- architecture;
- training code;
- evaluation;
- inference;
- data processing;
- deployment logic.

---

# License

This project is released under the MIT License. See `LICENSE`.
