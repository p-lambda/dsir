from pathlib import Path
import argparse
import random
import shutil
from json import loads, dumps
from tqdm import tqdm
from datasets import load_dataset
from glob import glob
from multiprocessing import Pool, cpu_count
import numpy as np
from utils import *
from time import time
import os


def compute_ngrams_raw(args, in_path: str, cache_path: Path):
    file_name = in_path.split("/")[-1]
    save_path = cache_path / f'{file_name}_{args.ngrams}grams_buckets_{args.num_buckets}_all.npy'

    if not save_path.exists():
        num_docs = 0
        st = time()
        counts = np.zeros(args.num_buckets).astype(int)
        with open(in_path, 'r') as f:
            for line in f:
                ex = loads(line)
                num_docs += 1
                if num_docs % 10000 == 0:
                    speed = num_docs / (time() - st)
                    print(num_docs, file_name, speed)
                line = ex["text"]
                curr_count = get_ngram_info(line, n=args.ngrams, num_buckets=args.num_buckets)
                counts = counts + curr_count
        np.save(str(save_path), counts)
    else:
        counts = np.load(str(save_path))
    print(file_name, "done!")
    return counts


def compute_importance_weights(args, in_path: str):
    file_name = in_path.split("/")[-1]
    out_dir = Path(args.out_path) / f'logratio_{args.ngrams}grams_buckets_{args.num_buckets}'
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f'{file_name}.npy'

    if not save_path.exists():
        logratios = []
        st = time()
        num_docs = 0
        with open(in_path, 'r') as f:
            for line in f:
                ex = loads(line)
                line = ex["text"]
                num_docs += 1
                if num_docs % 10000 == 0:
                    speed = num_docs / (time() - st)
                    print(num_docs, file_name, speed)
                curr_count = get_ngram_info(line, n=args.ngrams, num_buckets=args.num_buckets)
                logratio = np.inner(curr_count, args.log_diff_dist)
                logratios.append(logratio)
            logratios = np.asarray(logratios)
        np.save(str(save_path), logratios)
    else:
        logratios = np.load(str(save_path))
    print(file_name, "log ratio done!")
    return logratios


def resample(args, data_files, cache_ds_dir, streaming=False):
    retrieved_dir = args.out_path / f'retrieved'
    retrieved_path_cache = cache_ds_dir / f'retrieved_{args.num_to_retrieve}.jsonl'

    retrieved_dir.mkdir(parents=True, exist_ok=True)
    retrieved_path = retrieved_dir / f'{args.ds_name}_{args.target_ds_name}_retrieved_{args.num_to_retrieve}.jsonl'

    # merge logratio chunks
    logratios_file = retrieved_dir / 'logratios.npy'

    if not logratios_file.exists():
        logratios = []
        chunk_dir = f'{args.out_path}/logratio_{args.ngrams}grams_buckets_{args.num_buckets}'
        all_logratio_files = sorted(glob(f"{chunk_dir}/*.npy"))
        for curr_logratio_file in all_logratio_files:
            logratios.append(np.load(str(curr_logratio_file)))
        logratios = np.concatenate(logratios)
        np.save(logratios_file, logratios)
    else:
        logratios = np.load(logratios_file)

    print("logratios cnt", len(logratios))

    # noise the logratios
    logratios += np.random.gumbel(size=len(logratios))

    # choose top k
    chosen_idxs = np.argpartition(-logratios, args.num_to_retrieve)[:args.num_to_retrieve]

    global_mask = np.zeros(len(logratios)).astype(bool)
    global_mask[chosen_idxs] = True

    del nonzero_idxs
    del chosen_idxs
    del logratios

    print("Loading data...")

    # data_files = sorted(glob(f"{args.data_pool_path}/*.jsonl"))
    if not streaming:
        combined_streaming_ds = []
        for f_name in data_files:
            with open(f_name, "r") as f:
                combined_streaming_ds.extend(f.read().splitlines())
        print("data line cnt", len(combined_streaming_ds))

        assert len(combined_streaming_ds) == len(global_mask)

        with open(retrieved_path_cache, 'w') as fout:
            for i, curr_ex in tqdm(enumerate(combined_streaming_ds)):
                if global_mask[i]:
                    fout.write(curr_ex.strip() + '\n')
    else:
        combined_streaming_ds = load_dataset(
            'json',
            data_files=data_files,
            streaming=True)["train"]

        with open(retrieved_path_cache, 'w') as fout:
            for i, curr_ex in tqdm(enumerate(combined_streaming_ds)):
                # curr_ex["timestamp"] = curr_ex["timestamp"].strftime("%m/%d/%Y, %H:%M:%S")
                if global_mask[i]:
                    fout.write(dumps(curr_ex) + "\n")

    shutil.move(retrieved_path_cache, retrieved_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data selection with DSIR')
    parser.add_argument('--data_pool_path',
                        default="",
                        type=str,
                        help='path to data pool')
    parser.add_argument('--ds_name', default="data pool", type=str, help='pretraining dataset name')
    parser.add_argument('--target_path',
                        default="",
                        type=str,
                        help='path to target data')
    parser.add_argument('--target_ds_name', default="target data", type=str, help='target dataset name')
    parser.add_argument('--output_dir', default="", type=str,
                        help='output path')
    parser.add_argument('--num_to_retrieve', type=int, default=200000, help='Number of examples to retrieve')
    parser.add_argument('--cache_dir', default="", type=str,
                        help='cache directory for datasets')
    parser.add_argument('--ngrams', type=int, default=3, help='N in N-grams. 2 means both unigram and bigram.')
    parser.add_argument('--num_buckets', type=int, default=10000, help='number of ngram hash buckets')
    parser.add_argument('--pipeline_step', default="resample", type=str,
                        help='which step of pipeline to run. (importance_weights, resample)')
    args = parser.parse_args()
    random.seed(42)

    print("PYTHONHASHSEED", os.environ.get('PYTHONHASHSEED'))

    cache_ds_dir = Path(args.cache_dir) / 'ngram_cache' / args.ds_name
    cache_ds_dir.mkdir(exist_ok=True, parents=True)

    cache_target_dir = Path(args.cache_dir) / 'ngram_cache' / args.target_ds_name
    cache_target_dir.mkdir(exist_ok=True, parents=True)

    args.out_path = Path(args.output_dir) / args.target_ds_name / args.ds_name
    args.out_path.mkdir(exist_ok=True, parents=True)

    ds_in_paths = sorted(glob(f"{args.data_pool_path}/*"))
    target_in_paths = sorted(glob(f"{args.target_path}/*"))
    with Pool(cpu_count()) as p:
        ds_all_args = [(args, in_path, cache_ds_dir) for in_path in ds_in_paths]
        ds_ngram_dist = p.starmap(compute_ngrams_raw, ds_all_args)

        target_all_args = [(args, in_path, cache_target_dir) for in_path in target_in_paths]
        target_ngram_dist = p.starmap(compute_ngrams_raw, target_all_args)
    for i in range(1, len(ds_ngram_dist)):
        ds_ngram_dist[0] = ds_ngram_dist[0] + ds_ngram_dist[i]
        ds_ngram_dist[i] = None
    ds_ngram_dist = ds_ngram_dist[0]
    ds_ngram_dist = ds_ngram_dist / ds_ngram_dist.sum()

    for i in range(1, len(target_ngram_dist)):
        target_ngram_dist[0] = target_ngram_dist[0] + target_ngram_dist[i]
        target_ngram_dist[i] = None
    target_ngram_dist = target_ngram_dist[0]
    target_ngram_dist = target_ngram_dist / target_ngram_dist.sum()
    print(ds_ngram_dist)
    print(target_ngram_dist)

    args.log_diff_dist = np.log(target_ngram_dist + 1e-8) - np.log(ds_ngram_dist + 1e-8)
    with Pool(cpu_count()) as p:
        importance_weights_args = [(args, in_path) for in_path in ds_in_paths]
        importance_weights = p.starmap(compute_importance_weights, importance_weights_args)

    resample(args, ds_in_paths, cache_ds_dir)
