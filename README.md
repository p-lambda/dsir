# Data Selection for Language Models via Importance Resampling (DSIR)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2305.10429-00ff00.svg)](https://arxiv.org/abs/2302.03169)

This repository contains the [DSIR](https://arxiv.org/abs/2302.03169) data selection tool for selecting relevant language model training data from any raw data source given a target dataset, as well as pre-filtered datasets and some pretrained models.

DSIR is built for:
- fast, large-scale (trillion-token scale) data selection from large raw text datasets (Pile, RefinedWeb, RedPajama, ...). There is almost no overhead to selecting more examples (unlike retrieval), other than the time it takes to write the extra examples to disk.
- selecting data that is distributed like a given target dataset (domain-specific data, Wikipedia, ...). Relevance and diversity are balanced automatically by matching the distribution of the target dataset on a feature space (e.g., n-gram frequencies).

Compute needed:
- 1 CPU node
- a decent amount of RAM (at least 64GB for most large datasets - need to hold a few floats per example in memory)
- a high number of cores. The data selection speed scales linearly with the CPU cores.

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

To select data, simply initialize a `HashedNgramDSIR` object and call the following functions:
```
from data_selection import HashedNgramDSIR

raw_datasets = [<list of paths>]
target_datasets = [<list of paths>]

dsir = HashedNgramDSIR(raw_datasets, target_datasets, cache_dir='/scr/dsir_cache')
dsir.fit_importance_estimator(num_tokens_to_fit='auto')
dsir.compute_importance_weights()
dsir.resample(out_dir='resampled', num_to_sample=10000000, cache_dir='/scr/resampled_cache')
```
Running this would write 10M documents in `jsonl` files inside an output directory named `resampled`. The files will first be written to `cache_dir` and moved to `out_dir` upon completion (set `cache_dir` to `None` to skip this step). For best performance, use uncompressed `jsonl` files stored on local file storage for all data paths and use as many CPU cores as possible, which allows each file to be virtually sharded across multiple cores. Custom functions for reading the data paths and extracting the text field from each example can be provided via the
`{raw,target}_load_dataset_fn` and `{raw,target}_parse_example_fn` arguments to the constructor. The number of tokens to use for fitting the importance weight estimator can be tuned with the `num_tokens_to_fit` argument (set to `all` to fit on full dataset). Top-k retrieval instead of sampling without replacement (the default) can be done by specifying `top_k=True` to the `resample` method.
 
For target datasets with a mixture of code and natural language, consider splitting up the code and natural language into separate target distributions (and resampling once with respect to each target) for best performance.

The `dsir` intermediate results (after `fit_importance_estimator` and `compute_importance_weights`) can be saved and loaded for later use, for example to resample 100M documents instead:
```
dsir.save('dsir_params_dir')

# later on
dsir.load('dsir_params_dir')
dsir.resample(out_dir='resampled', num_to_sample=100000000, cache_dir='/scr/resampled_cache')
```
The `save` method can be called at any time to save partial results.

## Speed benchmark on The Pile
Using 1 CPU node with 96GB RAM and 96 cores, we can select data from the full (decompressed) Pile dataset in less than *4.5 hours*.
The Pile dataset was first decompressed and placed onto the node's local file storage. The breakdown of timings for each step are:
- *Fit importance estimator* (with `num_tokens_to_fit="auto"`): 59.28 seconds
- *Compute importance weights*: 4.36 hours
- *Resample 10M documents* (with `cache_dir=None` and `out_dir` is a local storage location): 353.68 seconds
- *Total*: 4.47 hours

Subsequent resampling with the same target data is very cheap, and the runtime does not scale with the number of documents to select (unlike retrieval). Resampling 100M documents takes the same amount of time (less than *6 minutes*) as resampling 10M documents:
- *Resample 10M documents*: 353.68 seconds
- *Resample 100M documents*: 352.69 seconds

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

