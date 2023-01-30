# Data Selection for Language Models via Importance Resampling (DSIR)

This repository contains pre-filtered datasets and code for selecting relevant language model training data from The Pile.

## Pre-filtered datasets
We provide datasets on HuggingFace that are already pre-filtered from The Pile. Stay tuned for more datasets!

### DSIR-filtered-pile-50M
- Target distribution: Wikipedia, Books3, BookCorpus2, Gutenberg
- Raw dataset: The Pile
- The dataset contains 51.2M examples, most of which are selected from Pile subsets that are not Wikipedia or books-related. 4% of the data is selected from Wikipedia and books. Every example concatenates 2 snippets, possibly from different sources, to ensure that the examples are long enough for longer context models (512 or 1024 tokens). Metadata about which sources the text comes from is included with every example.
- Available on huggingface at https://huggingface.co/datasets/stanford-crfm/DSIR-filtered-pile-50M. Use with HuggingFace Datasets:
```
from datasets import load_dataset
dataset = load_dataset("stanford-crfm/DSIR-filtered-pile-50M")
```
- Comparisons for training BERT-style models:

| GLUE dev                                          |  MNLI |  QNLI |   QQP |   RTE | SST2 |  MRPC |  CoLA | STSB |   Avg |
|---------------------------------------------------|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| Random selection                                  | 82.63 |  86.9 | 89.57 | 67.37 | 90.05 | 87.40 | 49.41 | 88.63 | 80.25 |
| Heuristic classification (GPT-3/Pile/PaLM method) | 82.69 | 85.95 | 89.77 | 68.59 | 88.94 | 86.03 | 48.17 | 88.62 | 79.85 |
| DSIR                                              | 83.07 | 89.11 | 89.80 | 75.09 | 90.48 | 87.70 | 54.00 | 89.17 | 82.30 |



## Code




