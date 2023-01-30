from pathlib import Path
import json
import numpy as np
import os
from itertools import islice
from joblib import Parallel, delayed
from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm
from multiprocessing import Pool

import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

stop = set(stopwords.words('english') + list(string.punctuation))
numeric = set(list(string.digits))


def transform_text(text):
    return ' '.join(word_tokenize(text.lower()))


def length_filter(x_tok):
    return len(x_tok)


def repeating_filter(x_tok):
    if len(x_tok) == 0:
        return 0
    counts = Counter(x_tok)
    ratio = (max(counts.values()) / len(x_tok))
    return ratio


def mostly_uninformative_filter(x_tok):
    if len(x_tok) == 0:
        return 0
    informative_ratio = (len([x for x in x_tok if x not in stop]) / len(x_tok))
    return informative_ratio


def numeric_filter(x_tok):
    if len(x_tok) == 0:
        return 0
    ratio = (len([x for x in x_tok if x not in numeric]) / len(x_tok))
    return ratio


filter_funcs = [length_filter, repeating_filter, mostly_uninformative_filter, numeric_filter]


def process(example):
    line_json = json.loads(example)
    text_tok = transform_text(line_json['contents']).split()
    return [fn(text_tok) for fn in filter_funcs]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='compute statistic for quality filtering')
    parser.add_argument('--ds_path', type=str, help='path to jsonl dataset')
    parser.add_argument('--chunksize', type=int, default=100000, help='chunk size')
    parser.add_argument('--no_parallel', action='store_true', help='dont do in parallel')
    args = parser.parse_args()

    ds_path = Path(args.ds_path)
    num_cpus = len(os.sched_getaffinity(0))

    print(f"Num cpus: {num_cpus}")

    if args.no_parallel:
        scores = []
        with open(ds_path, 'r') as f:
            for line in f:
                processed = process(line)
                scores.append(processed)
                if len(scores) % 100000 == 0:
                    print(len(scores), flush=True)

    else:
        pool = Pool(num_cpus)
        chunk_size = args.chunksize
        scores = []
        with open(ds_path, 'r') as f:
            for processed in pool.imap(process, f, chunksize=chunk_size):
                scores.append(processed)
                if len(scores) % 100000 == 0:
                    print(len(scores), flush=True)
        pool.close()
    scores = np.asarray(scores)

    np.savez(str(ds_path.parent / (ds_path.name + '_qualityscores.npz')),
             length=scores[:, 0],
             repeating_ratio=scores[:, 1],
             informative_ratio=scores[:, 2],
             numeric_ratio=scores[:, 3])
