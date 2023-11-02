# Data Selection for Language Models via Importance Resampling (DSIR)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2305.10429-00ff00.svg)](https://arxiv.org/abs/2302.03169)

This repository contains the [DSIR](https://arxiv.org/abs/2302.03169) data selection tool for selecting relevant language model training data from any raw data source given a target dataset, as well as pre-filtered datasets and some pretrained models.

DSIR is built for:
- fast, large-scale (trillion-token scale) data selection from large raw text datasets (Pile, RefinedWeb, RedPajama, ...)
- selecting data that is distributed like a given target dataset (domain-specific data, Wikipedia, ...). Relevance and diversity are balanced automatically.

Compute needed:
- 1 CPU node
- a large amount of RAM (at least few hundred GB)
- a high number of cores (parallelism on a file level. For best performance, use as many CPU cores as data files)

![DSIR figure](fig1.png)

Code related to the DSIR paper's experiments are in the `experimental/` directory.

## Quickstart

Install from pip:
```
pip install data-selection
```

Install from source by cloning this repo and installing via pip:
```
git clone git@github.com:/p-lambda/dsir
pip install ./dsir
```

To select data, initialize a `HashedNgramDSIR` object and call the following functions:
```
from data_selection import HashedNgramDSIR

raw_datasets = [<list of paths>]
target_datasets = [<list of paths>]

dsir = HashedNgramDSIR(raw_datasets, num_proc=30)
dsir.fit_importance_estimator(target_datasets)
dsir.compute_importance_weights()
dsir.resample(out_dir='resampled', num_to_sample=1000000, cache_dir='/scr/resampled_cache')
```
This will save 1M examples in `jsonl` files inside an output directory named `resampled`. The files will first be written to `cache_dir` and moved to `out_dir` upon completion.

The `dsir` intermediate results (after `fit_importance_estimator` and `compute_importance_weights`) can be saved and loaded for later use, for example to resample a different number of examples:
```
dsir.save('dsir_params')

# later on
dsir.load('dsir_params')
dsir.resample(out_dir='resampled', num_to_sample=10000000, cache_dir='/scr/resampled_cache')
```

## Pre-filtered datasets
Note: previous versions of the datasets had a small validation and test split (50000 examples each), but we concatenated these onto the end of the train set (in the order validation, then test) to better align with the paper. The datasets should be further shuffled during preprocessing before training.

### DSIR-filtered-pile-50M
- Target distribution: Wikipedia, BookCorpus2
- Selection method: DSIR (with importance resampling on hashed n-gram model importance weights)
- Raw dataset: The Pile
- Size: 80GB, 51.2M examples
- Used for 128-token context models in the paper. Suitable for token length 512 or 1024, but can be used for shorter token lengths.
- The dataset contains 51.2M examples, most of which are selected from Pile subsets that are not Wikipedia or books-related (BookCorpus2, Books3, Gutenberg). 4% of the data is randomly selected from Wikipedia and books-related subsets. Every example concatenates 2 snippets, possibly from different sources, to ensure that the examples are long enough for longer context models (512 or 1024 tokens). Metadata about which sources the text comes from is included with every example.
- Available on HuggingFace at https://huggingface.co/datasets/stanford-crfm/DSIR-filtered-pile-50M. Use with HuggingFace Datasets:
```
from datasets import load_dataset
dataset = load_dataset("stanford-crfm/DSIR-filtered-pile-50M")
```

### heuristic_classification-filtered-pile-50M
- Target distribution: Wikipedia, BookCorpus2
- Selection method: Heuristic classification (FastText binary classifier)
- Raw dataset: The Pile
- Size: 80GB, 51.2M examples
- Used for 128-token context length models in the paper. Suitable for token length 512 or 1024, but can be used for shorter token lengths
- The dataset contains 51.2M examples, most of which are selected from Pile subsets that are not Wikipedia or books-related (BookCorpus2, Books3, Gutenberg). 4% of the data is randomly selected from Wikipedia and books-related subsets. Every example concatenates 2 snippets, possibly from different sources, to ensure that the examples are long enough for longer context models (512 or 1024 tokens). Metadata about which sources the text comes from is included with every example.
- Available on HuggingFace at https://huggingface.co/datasets/stanford-crfm/heuristic_classification-filtered-pile-50M. Use with HuggingFace Datasets:
```
from datasets import load_dataset
dataset = load_dataset("stanford-crfm/heuristic_classification-filtered-pile-50M")
```
- Comparisons for training BERT-base models from scratch (50k steps, 128 max token length, 4096 batch size):

| GLUE dev                                          |  MNLI |  QNLI |   QQP |   RTE | SST2 |  MRPC |  CoLA | STSB |   Avg |
|---------------------------------------------------|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| Random selection from The Pile                    | 82.63 |  86.9 | 89.57 | 67.37 | 90.05 | 87.40 | 49.41 | 88.63 | 80.25 |
| Heuristic classification (GPT-3/Pile/PaLM method) | 82.69 | 85.95 | 89.77 | 68.59 | 88.94 | 86.03 | 48.17 | 88.62 | 79.85 |
| DSIR                                              | 83.07 | 89.11 | 89.80 | 75.09 | 90.48 | 87.70 | 54.00 | 89.17 | 82.30 |


## Pretrained models

In the table below, `{dataset}` can be replaced with one of `{ag, amazon, citation_intent, hyp, imdb, sciie, chemprot, rct-20k}` for the continued pretraining models.

| HuggingFace ID | Link | Dataset size | Max token length | Training steps | Architecture | Initialization | Description |
|---|---|---|---|---|---|---|---|
| dsir-bert-scratch-wiki_and_books | [Link](https://huggingface.co/sangmichaelxie/dsir-bert-scratch-wiki_and_books) | 6.5B tokens (51.2M examples) | 128 | 5.00E+04 | bert-base-uncased | scratch | BERT model trained on [DSIR-filtered-pile-50M](https://huggingface.co/datasets/stanford-crfm/DSIR-filtered-pile-50M/viewer/default/train?p=31445&row=3144531) |
| heuristiccls-bert-scratch-wiki_and_books | [Link](https://huggingface.co/sangmichaelxie/heuristiccls-bert-scratch-wiki_and_books) | 6.5B tokens (51.2M examples) | 128 | 5.00E+04 | bert-base-uncased | scratch | BERT model trained on Pile data filtered by heuristic classification |
| randomselect-bert-scratch | [Link](https://huggingface.co/sangmichaelxie/randomselect-bert-scratch) | 6.5B tokens (51.2M examples) | 128 | 5.00E+04 | bert-base-uncased | scratch | BERT model trained on random subset of The Pile |
| dsir-roberta-continuedpretrain-{dataset} | Link format: `https://huggingface.co/sangmichaelxie/dsir-roberta-continuedpretrain-{dataset}` | 6.4B tokens (25M examples) | 256 | 25000 | roberta-base | roberta-base | RoBERTa model with continued pretraining on data selected by DSIR with target={dataset} |
| heuristiccls-roberta-continuedpretrain-{dataset} | Link format: `https://huggingface.co/sangmichaelxie/dsir-roberta-continuedpretrain-{dataset}` | 6.4B tokens (25M examples) | 256 | 25000 | roberta-base | roberta-base | RoBERTa model with continued pretraining on data selected by heurstic classification with target={dataset} |
| randomselect-roberta-continuedpretrain | [Link](https://huggingface.co/sangmichaelxie/randomselect-roberta-continuedpretrain) | 6.4B tokens (25M examples) | 256 | 25000 | roberta-base | roberta-base | RoBERTa model with continued pretraining on random subset of The Pile |

## Citation Information
Paper: <https://arxiv.org/abs/2302.03169>
```
@article{xie2023data,
  author = {Sang Michael Xie and Shibani Santurkar and Tengyu Ma and Percy Liang},
  journal = {Advances in Neural Information Processing Systems (NeurIPS)},
  title = {Data Selection for Language Models via Importance Resampling},
  year = {2023},
}
```

