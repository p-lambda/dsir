import pandas as pd
from pathlib import Path
from collections import defaultdict
import json
import subprocess

task_to_col = {'QNLI': 'eval_accuracy',
               'STSB': 'eval_spearmanr',
               'MRPC': 'eval_accuracy',
               'COLA': 'eval_matthews_correlation',
               'RTE': 'eval_accuracy',
               'MNLI': 'eval_accuracy',
               'SST2': 'eval_accuracy',
               'QQP': 'eval_accuracy'}

def read_file(path, task_name):
    with open(path, 'r') as f:
        curr_res = json.load(f)
    eval_res = curr_res[task_to_col[task_name]]
    del curr_res[task_to_col[task_name]]
    curr_res['dev'] = eval_res
    return curr_res

def parse_file_name(name):
    toks = name.split('_')
    task_name = toks[0]

    for tok in toks[1:]:
        if tok.startswith('EPOCHS'):
            epochs = int(tok[6:])
        elif tok.startswith('BS'):
            bs = int(tok[2:])
        elif tok.startswith('LR'):
            lr = float(tok[2:])
        elif tok.startswith('seed'):
            seed = int(tok[4:])
    return {
        'task_name': task_name,
        'epochs': epochs,
        'bs': bs,
        'lr': lr,
        'seed': seed, }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Read GLUE results')
    parser.add_argument('--results_dir', type=str, help="directory of GLUE results")
    args = parser.parse_args()

    res = []

    results_dir = Path(args.results_dir).resolve().expanduser()
    for trial_dir in results_dir.iterdir():
        if not trial_dir.is_dir():
            continue
        curr_res = parse_file_name(trial_dir.name)

        try:
            curr_res.update(read_file(trial_dir / f'eval_results.json', curr_res['task_name']))
            res.append(curr_res)
        except Exception:
            print(f"skipped: {curr_res}")

    df = pd.DataFrame(res)
    df['dev'] *= 100
    df = df.round({'dev': 2})
    grouped_df = df.groupby(['task_name', 'epochs', 'bs', 'lr']).agg({'dev': ['mean', 'std', 'median']})
    grouped_df.columns = ['mean', 'std', 'median']
    grouped_df = grouped_df.reset_index()
    # we only run one set of hyperparams per task in the paper, so this max-median selection is a no-op
    max_median_idx = grouped_df.groupby(['task_name', 'epochs', 'bs', 'lr'])['median'].transform(max) == grouped_df['median']
    df = grouped_df[max_median_idx]
    df = df.reset_index()
    df = df.round(2)
    df.to_csv(str(results_dir / 'glue_results_eval.tsv'), sep='\t', index=False)

    subprocess.run(f"head -n 50 {str(results_dir / 'glue_results_eval.tsv')}".split())

