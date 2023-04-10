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



## Code for data selection

To select your own subset of The Pile, all you need is a small set of target examples representing the kind of data you want to select.
This target dataset should be in jsonl format -- it can also be a dataset from HuggingFace Datasets. Note that our current workflow requires about 2TB of storage space --- we're working on reducing this!
1. Create a virtualenv using `requirements.txt`: `virtualenv .venv; source .venv/bin/activate; pip install -r requirements.txt`
2. Download The Pile to `PILE_PATH` and change the corresponding variables in `config.sh`.
3. Run preprocessing on The Pile: Go to `preprocessing/` and run `run_slurm.sh`. You can also use `run.sh` directly with the arguments from the Slurm command. This only needs to be run once. 
4. Precompute quality filter stats: Go to `preprocessing/quality_scores/` and run `run_slurm_quality_stats.sh`. After this, run `merge_quality_scores.py`. This only needs to be run once. (We're working on streamlining steps 3 and 4. Stay tuned!) 
5. Run DSIR: Go to `data_selection/`. An example is in `run_cmds.sh`. For new target datasets, some information about which fields in the dataset to use should be placed in the `dsname_to_args` dictionary at the top of the `dsir_pipeline.py` file. If you wish to retrieve from custom subsets of the Pile, you will need to tweak one part of the code, in the main part of the script (an example is provided of how to do so). Many of the steps in DSIR can be cached and will only run the first time. For example, resampling a different number of examples with the same target dataset uses cached importance weights.

## Code for pretraining and GLUE evaluation

We provide scripts for training BERT-style masked language models on the selected data and evaluating it on GLUE in the `train` and `glue_eval` directories, respectively.
1. Install further dependencies using `train/requirements.txt`: `pip install -r train/requirements.txt`
2. Change the `PRETRAIN_OUTPUT_DIR` variable in `config.sh`.
3. Write a job command in `train/run_slurm.sh`. An example command in this file. You will need to change the path to the training data. If you want to skip preprocessing (if it's already done), set the first of two boolean variables to `false`. By setting both to `true`, there will be two jobs launched: one for preprocessing and one for pretraining. The pretraining job should take about 50 hours on 4 RTX 3090 GPUs. Kick off the jobs by running `cd train; bash run_slurm.sh`.
4. Evaluate the trained model by editing the evaluation job command in `glue_eval/run_eval_exps.sh` with the path to the model checkpoint. This script runs 5 seeds for each GLUE dataset. The results and finetuned models will be saved a new `finetune_runs` directory inside the pretrained model checkpoint directory. Kick off the jobs by running `cd glue_eval; bash run_eval_exps.sh`.
5. Read the GLUE results by running `python read_glue_results.py --results_dir </path/to/checkpoint>/finetune_runs` in the `glue_eval` directory.

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

