---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: 'Main product: Motorola Edge 60 Neo 5G. Review: While the 6000maH Si-C battery
    can be retained from the Edge 60 Pro and also a bigger and better vapor cooling
    chamber'
- text: 'Main product: Samsung Galaxy S21+ 5G. Review: Pros The best OLED screen,
    1000+nits, 1440p, 120Hz, HDR10+, S-Pen Phenomenal fingerprint scanner performance
    Outstanding battery life, fast to top-up the 5, 000mAh battery Stereo speakers
    with good loudness'
- text: 'Main product: Samsung Galaxy S20+. Review: I too had the same issue. It started
    with discoloration and ended with screen burn in. And this was 11 months in. Luckily,
    I got the screen replaced under warranty.'
- text: 'Main product: Xiaomi Poco F7 5G. Review: 9, 10, Pros and Cons:, Pro:, Performance
    (Snapdragon 8 Elite), Battery life (e g Genshin impact ran for around 3 hours
    at highest graphics settings), Design looks unique and different from other phones,'
- text: 'Main product: Samsung Galaxy S23 Ultra. Review: The device is now stuck at
    the Service Center since March 23rd, and the technicians dont have guaranteed
    non-faulty camera modules to make a repair even if they wanted to'
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
base_model: sentence-transformers/paraphrase-mpnet-base-v2
---

# SetFit with sentence-transformers/paraphrase-mpnet-base-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 512 tokens
<!-- - **Number of Classes:** Unknown -->
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("Main product: Motorola Edge 60 Neo 5G. Review: While the 6000maH Si-C battery can be retained from the Edge 60 Pro and also a bigger and better vapor cooling chamber")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median  | Max  |
|:-------------|:----|:--------|:-----|
| Word count   | 16  | 57.5688 | 2764 |

### Training Hyperparameters
- batch_size: (8, 8)
- num_epochs: (1, 1)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 10
- body_learning_rate: (2e-05, 2e-05)
- head_learning_rate: 2e-05
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0001 | 1    | 0.1275        | -               |
| 0.0032 | 50   | 0.1439        | -               |
| 0.0064 | 100  | 0.1141        | -               |
| 0.0096 | 150  | 0.0768        | -               |
| 0.0129 | 200  | 0.0666        | -               |
| 0.0161 | 250  | 0.0446        | -               |
| 0.0193 | 300  | 0.0413        | -               |
| 0.0225 | 350  | 0.0364        | -               |
| 0.0257 | 400  | 0.032         | -               |
| 0.0289 | 450  | 0.029         | -               |
| 0.0321 | 500  | 0.0271        | -               |
| 0.0353 | 550  | 0.022         | -               |
| 0.0386 | 600  | 0.0267        | -               |
| 0.0418 | 650  | 0.0196        | -               |
| 0.0450 | 700  | 0.0243        | -               |
| 0.0482 | 750  | 0.0217        | -               |
| 0.0514 | 800  | 0.0218        | -               |
| 0.0546 | 850  | 0.017         | -               |
| 0.0578 | 900  | 0.0201        | -               |
| 0.0610 | 950  | 0.0172        | -               |
| 0.0643 | 1000 | 0.0229        | -               |
| 0.0675 | 1050 | 0.0207        | -               |
| 0.0707 | 1100 | 0.0175        | -               |
| 0.0739 | 1150 | 0.022         | -               |
| 0.0771 | 1200 | 0.0166        | -               |
| 0.0803 | 1250 | 0.0148        | -               |
| 0.0835 | 1300 | 0.0169        | -               |
| 0.0867 | 1350 | 0.0166        | -               |
| 0.0900 | 1400 | 0.0156        | -               |
| 0.0932 | 1450 | 0.016         | -               |
| 0.0964 | 1500 | 0.0202        | -               |
| 0.0996 | 1550 | 0.0182        | -               |
| 0.1028 | 1600 | 0.0134        | -               |
| 0.1060 | 1650 | 0.0202        | -               |
| 0.1092 | 1700 | 0.0152        | -               |
| 0.1124 | 1750 | 0.0152        | -               |
| 0.1157 | 1800 | 0.0139        | -               |
| 0.1189 | 1850 | 0.0161        | -               |
| 0.1221 | 1900 | 0.0155        | -               |
| 0.1253 | 1950 | 0.0133        | -               |
| 0.1285 | 2000 | 0.0137        | -               |
| 0.1317 | 2050 | 0.0173        | -               |
| 0.1349 | 2100 | 0.01          | -               |
| 0.1381 | 2150 | 0.0114        | -               |
| 0.1414 | 2200 | 0.0141        | -               |
| 0.1446 | 2250 | 0.0142        | -               |
| 0.1478 | 2300 | 0.0115        | -               |
| 0.1510 | 2350 | 0.0117        | -               |
| 0.1542 | 2400 | 0.0115        | -               |
| 0.1574 | 2450 | 0.0105        | -               |
| 0.1606 | 2500 | 0.0107        | -               |
| 0.1639 | 2550 | 0.0136        | -               |
| 0.1671 | 2600 | 0.0166        | -               |
| 0.1703 | 2650 | 0.0092        | -               |
| 0.1735 | 2700 | 0.01          | -               |
| 0.1767 | 2750 | 0.0083        | -               |
| 0.1799 | 2800 | 0.0137        | -               |
| 0.1831 | 2850 | 0.0098        | -               |
| 0.1863 | 2900 | 0.0122        | -               |
| 0.1896 | 2950 | 0.0069        | -               |
| 0.1928 | 3000 | 0.0097        | -               |
| 0.1960 | 3050 | 0.0109        | -               |
| 0.1992 | 3100 | 0.0174        | -               |
| 0.2024 | 3150 | 0.014         | -               |
| 0.2056 | 3200 | 0.0099        | -               |
| 0.2088 | 3250 | 0.0076        | -               |
| 0.2120 | 3300 | 0.0098        | -               |
| 0.2153 | 3350 | 0.0083        | -               |
| 0.2185 | 3400 | 0.0075        | -               |
| 0.2217 | 3450 | 0.0116        | -               |
| 0.2249 | 3500 | 0.0156        | -               |
| 0.2281 | 3550 | 0.0079        | -               |
| 0.2313 | 3600 | 0.0093        | -               |
| 0.2345 | 3650 | 0.0114        | -               |
| 0.2377 | 3700 | 0.012         | -               |
| 0.2410 | 3750 | 0.0107        | -               |
| 0.2442 | 3800 | 0.0089        | -               |
| 0.2474 | 3850 | 0.0053        | -               |

### Framework Versions
- Python: 3.12.12
- SetFit: 1.1.3
- Sentence Transformers: 5.1.1
- Transformers: 4.57.1
- PyTorch: 2.8.0+cu126
- Datasets: 4.0.0
- Tokenizers: 0.22.1

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->