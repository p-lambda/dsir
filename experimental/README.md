# Code for the DSIR paper
This directory has the code for preprocessing, data selection, pretraining, and fine-tuning for the experiments in the DSIR paper. Pre-filtered datasets and pre-trained models from the paper are linked in the README at the outer directory.

## Code for data selection

To select your own subset of The Pile, all you need is a small set of target examples representing the kind of data you want to select.
This target dataset should be in jsonl format -- it can also be a dataset from HuggingFace Datasets. Note that our current workflow requires about 2TB of storage space --- we're working on reducing this! All the code should be run from the `experimental/` directory.
1. Create a virtualenv using `requirements.txt`: `virtualenv .venv; source .venv/bin/activate; pip install -r requirements.txt`
2. Download The Pile to `PILE_PATH` and change the corresponding variables in `config.sh`.
3. Run preprocessing on The Pile: Run `bash preprocessing/run_slurm.sh`. You can also run `bash preprocessing/run.sh` directly using the arguments in `preprocessing/run_slurm.sh`. This only needs to be run once. 
4. Precompute quality filter stats: Run `bash preprocessing/quality_scores/run_slurm_quality_stats.sh`. After this, run `bash preprocessing/quality_scores/run_merge_quality_scores.sh`. This only needs to be run once. (We're working on streamlining steps 3 and 4. Stay tuned!) 
5. Run DSIR: For an example, run `bash data_selection/run_cmds.sh`. For new target datasets, some information about which fields in the dataset to use should be placed in the `dsname_to_args` dictionary at the top of the `data_selection/dsir_pipeline.py` file. If you wish to retrieve from custom subsets of the Pile (for example, only select data from one chunk of the Pile), you will need to tweak one part of the code, in the main part of the script (an example is provided of how to do so as a comment). Many of the steps in DSIR can be cached and will only run the first time. For example, resampling a different number of examples with the same target dataset uses cached importance weights.

## Code for pretraining and GLUE evaluation

We provide scripts for training BERT-style masked language models on the selected data and evaluating it on GLUE in the `train` and `glue_eval` directories, respectively. All code should be run from the `experimental/` directory.
1. Install further dependencies using `train/requirements.txt`: `pip install -r train/requirements.txt`
2. Change the `PRETRAIN_OUTPUT_DIR` variable in `config.sh`.
3. Write a job command in `train/run_slurm.sh`. An example command in this file. You will need to change the path to the training data. If you want to skip preprocessing (if it's already done), set the first of two boolean variables to `false`. By setting both to `true`, there will be two jobs launched: one for preprocessing and one for pretraining. The pretraining job should take about 50 hours on 4 RTX 3090 GPUs. Kick off the jobs by running `bash train/run_slurm.sh`.
4. Evaluate the trained model by editing the evaluation job command in `glue_eval/run_eval_exps.sh` with the path to the model checkpoint. This script runs 5 seeds for each GLUE dataset. The results and finetuned models will be saved a new `finetune_runs` directory inside the pretrained model checkpoint directory. Kick off the jobs by running `bash glue_exps/run_eval_exps.sh`.
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

