# Data Selection for Language Models via Importance Resampling (DSIR)

This repository contains pre-filtered datasets and code for selecting relevant language model training data from The Pile.

## Pre-filtered datasets
We provide datasets on HuggingFace that are already pre-filtered from The Pile. Stay tuned for more datasets!

### DSIR-filtered-pile-50M
- Target distribution: Wikipedia, BookCorpus2
- Raw dataset: The Pile
- Size: 80GB, 51.2M examples
- Suitable for token length 512 or 1024, but can be used for shorter token lengths
- The dataset contains 51.2M examples, most of which are selected from Pile subsets that are not Wikipedia or books-related (BookCorpus2, Books3, Gutenberg). 4% of the data is randomly selected from Wikipedia and books-related subsets. Every example concatenates 2 snippets, possibly from different sources, to ensure that the examples are long enough for longer context models (512 or 1024 tokens). Metadata about which sources the text comes from is included with every example.
- Available on HuggingFace at https://huggingface.co/datasets/stanford-crfm/DSIR-filtered-pile-50M. Use with HuggingFace Datasets:
```
from datasets import load_dataset
dataset = load_dataset("stanford-crfm/DSIR-filtered-pile-50M")
```
- Comparisons for training BERT-base models from scratch (50k steps, 128 max token length, 4096 batch size):

| GLUE dev                                          |  MNLI |  QNLI |   QQP |   RTE | SST2 |  MRPC |  CoLA | STSB |   Avg |
|---------------------------------------------------|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| Random selection from The Pile                    | 82.63 |  86.9 | 89.57 | 67.37 | 90.05 | 87.40 | 49.41 | 88.63 | 80.25 |
| Heuristic classification (GPT-3/Pile/PaLM method) | 82.69 | 85.95 | 89.77 | 68.59 | 88.94 | 86.03 | 48.17 | 88.62 | 79.85 |
| DSIR                                              | 83.07 | 89.11 | 89.80 | 75.09 | 90.48 | 87.70 | 54.00 | 89.17 | 82.30 |



## Code

To select your own subset of The Pile, all you need is a small set of target examples representing the kind of data you want to select.
This target dataset should be in jsonl format -- it can also be a dataset from HuggingFace Datasets. Note that our current workflow requires about 3TB of storage space --- we're working on reducing this!
1. Create a virtualenv using `requirements.txt`: `virtualenv .venv; source .venv/bin/activate; pip install -r requirements.txt`
2. Download The Pile to `PILE_PATH` and change the corresponding variables in `config.sh`.
3. Run preprocessing on The Pile: Go to `preprocessing/` and run `run_slurm.sh`. You can also use `run.sh` directly with the arguments from the Slurm command. This only needs to be run once. 
4. Precompute quality filter stats: Go to `preprocessing/quality_scores/` and run `run_slurm_quality_stats.sh`. This only needs to be run once. (We're working on removing/streamlining steps 3 and 4. Stay tuned!) 
5. Run DSIR: Go to `data_selection/`. An example is in `run_cmds.sh`. For new target datasets, some information about which fields in the dataset to use should be placed in the `dsname_to_args` dictionary at the top of the `is_pipeline.py` file. Many of the steps in DSIR can be cached and will only run the first time. For example, resampling a different number of examples with the same target dataset uses cached importance weights.

## Citation Information
Paper: <https://arxiv.org/abs/2302.03169>
```
@article{xie2023data,
  author = {Sang Michael Xie and Shibani Santurkar and Tengyu Ma and Percy Liang},
  journal = {arXiv preprint arXiv:2302.03169},
  title = {Data Selection for Language Models via Importance Resampling},
  year = {2023},
}
```

