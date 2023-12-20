from pathlib import Path
from itertools import zip_longest
import random
import argparse
import json
import shutil
from collections import defaultdict
import subprocess
from itertools import islice

from tqdm import tqdm
from nltk.tokenize import WordPunctTokenizer
import numpy as np
from datasets import load_dataset

import logging
logging.basicConfig(level=logging.INFO)


# Place information about datasets in the dict below.
# The columns field is a list of columns to use for DSIR.
dsname_to_args = {
    'ag_news': {'dataset_name': 'yxchar/ag-tlm',
                'task_name': None,
                'columns': ['text'], },
    'chemprot': {'dataset_name': "yxchar/chemprot-tlm",
                 'task_name': None,
                 'columns': ['text']},
    'citation_intent': {'dataset_name': "yxchar/citation_intent-tlm",
                        'task_name': None,
                        'columns': ['text']},
    'hyperpartisan': {'dataset_name': "yxchar/hyp-tlm",
                      'task_name': None,
                      'columns': ['text']},
    'rct': {'dataset_name': "yxchar/rct-20k-tlm",
            'task_name': None,
            'columns': ['text']},
    'imdb': {'dataset_name': 'yxchar/imdb-tlm',
             'task_name': None,
             'columns': ['text']},
    'sciie': {'dataset_name': 'yxchar/sciie-tlm',
              'task_name': None,
              'columns': ['text']},
    'helpfulness': {'dataset_name': 'yxchar/amazon-tlm',
                    'task_name': None,
                    'columns': ['text']},
    'pile_val': {'dataset_name': 'json',
                 'task_name': None,  # to be set later
                 'quality_scores': None,  # to be set later
                 'columns': ['contents'], },
    'pile': {'dataset_name': 'json',
             'task_name': None,  # to be set later
             'quality_scores': None,  # to be set later
             'columns': ['contents'],
             'total_lines': 1745766302, },
 }


subsets = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
           '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
           '21', '22', '23', '24', '25', '26', '27', '28', '29']


def get_quality_mask(quality_scores):
    keep = (
        (quality_scores['length'] > 40)
        & (quality_scores['length'] < 500)
        & (quality_scores['repeating_ratio'] > 0.02)
        & (quality_scores['repeating_ratio'] < 0.2)
        & (quality_scores['informative_ratio'] > 0.3)
        & (quality_scores['informative_ratio'] < 0.7)
        & (quality_scores['numeric_ratio'] > 0.8)
    )
    return keep


def hash_buckets(string, num_buckets=10e4):
    return int(abs(hash(string)) % num_buckets)


def unigrams_bigrams(text):
    words = text.split()
    return words, list(zip(words, islice(words, 1, None)))


wpt = WordPunctTokenizer()
def get_ngram_info(line, n=2, num_buckets=10000):
    words = wpt.tokenize(line.lower())
    unigrams, bigrams = words, list(zip(words, islice(words, 1, None)))

    counts = np.zeros(num_buckets, dtype=int)
    for unigram in unigrams:
        counts[hash_buckets(unigram, num_buckets=num_buckets)] += 1
    for bigram in bigrams:
        counts[hash_buckets(bigram, num_buckets=num_buckets)] += 1
    return counts


def grouper(iterable, n, *, incomplete='fill', fillvalue=None):
    "Collect data into non-overlapping fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, fillvalue='x') --> ABC DEF Gxx
    # grouper('ABCDEFG', 3, incomplete='strict') --> ABC DEF ValueError
    # grouper('ABCDEFG', 3, incomplete='ignore') --> ABC DEF
    args = [iter(iterable)] * n
    if incomplete == 'fill':
        return zip_longest(*args, fillvalue=fillvalue)
    if incomplete == 'strict':
        return zip(*args, strict=True)
    if incomplete == 'ignore':
        return zip(*args)
    else:
        raise ValueError('Expected fill, strict, or ignore')


def compute_ngrams_hf(ds_name, ds_dir, cache_dir, ngrams, num_buckets):
    ds_dir = Path(ds_dir)
    save_path = ds_dir / f'{ds_name}_ngramcounts.npy'
    if not save_path.exists():
        config = dsname_to_args[ds_name]
        text_cols = config["columns"]
        logging.info(f"{text_cols}")

        if ds_name == 'dbpedia':
            ds = load_dataset(config["dataset_name"], data_files=config["task_name"],
                              cache_dir=cache_dir, download_mode='force_redownload')
        else:
            ds = load_dataset(config["dataset_name"], config["task_name"],
                              cache_dir=cache_dir)

        counts = np.zeros(num_buckets).astype(int)
        for i, ex in tqdm(enumerate(ds['train']), miniters=10000, maxinterval=1000000):
            line = " ".join([ex[c] for c in text_cols])
            curr_count = get_ngram_info(line, n=ngrams)
            counts = counts + curr_count
        np.save(save_path, counts)
    else:
        counts = np.load(save_path)
    return counts


def compute_ngrams_pile(
        path, ngrams=2, num_buckets=10000,
        filter_domains=None,
        cache_dir=None):

    path_parent = Path(path).parent

    if filter_domains is None:
        save_path = path_parent / f'ngrams{ngrams}_buckets{num_buckets}_nofilter.npy'
    else:
        filter_domains_str = '_'.join(filter_domains)
        save_path = path_parent / f'ngrams{ngrams}_buckets{num_buckets}_{filter_domains_str}.npy'

    if not save_path.exists():

        counts = np.zeros(num_buckets).astype(int)
        num_docs = 0
        with open(path, 'r') as f:
            for k, line in tqdm(enumerate(f), miniters=1000000, maxinterval=1000000):
                ex = json.loads(line)
                domain = ex["metadata"]["pile_set_name"]
                if filter_domains is not None and domain not in filter_domains:
                    continue
                num_docs += 1
                line = ex["contents"]
                curr_count = get_ngram_info(line, n=ngrams)
                counts = counts + curr_count
        np.save(save_path, counts)
    else:
        counts = np.load(save_path)
    return counts


def compute_importance_weights(
        path, ds_dir, chunk_idx, target_dist, pile_dist, ngrams=2, num_buckets=10000):
    chunk_dir = Path(ds_dir) / f'logratio_chunks_ngrams{ngrams}_buckets{num_buckets}'
    chunk_dir.mkdir(parents=True, exist_ok=True)
    save_path = chunk_dir / f'{chunk_idx}.npy'

    log_diff_dist = np.log(target_dist + 1e-8) - np.log(pile_dist + 1e-8)

    if not save_path.exists():
        logratios = []
        with open(path, 'r') as f:
            for k, line in tqdm(enumerate(f), miniters=1000000, maxinterval=1000000):
                ex = json.loads(line)
                line = ex["contents"]
                curr_count = get_ngram_info(line, n=ngrams)
                logratio = np.inner(curr_count, log_diff_dist)
                logratios.append(logratio)
            logratios = np.asarray(logratios)
        np.save(save_path, logratios)
    else:
        logratios = np.load(save_path)
    return logratios


def compute_domain_idxs(filter_domains):
    # path to outer directory
    ds_path = Path(dsname_to_args['pile']['task_name'][0]).parent.parent

    domain_to_idxs = defaultdict(list)
    todo_domains = []
    for domain in filter_domains:
        domain_idxs_path = ds_path / f"{domain.replace(' ', '_')}_idxs.npy"
        if not domain_idxs_path.exists():
            todo_domains.append(domain)

    combined_streaming_ds = load_dataset(
            'json',
            data_files=dsname_to_args['pile']['task_name'],
            streaming=True)['train']

    todo_domains = set(todo_domains)
    if len(todo_domains) > 0:
        for i, ex in tqdm(enumerate(combined_streaming_ds), miniters=1000000):
            domain = ex["metadata"]["pile_set_name"]
            if domain in todo_domains:
                domain_to_idxs[domain].append(i)
        for domain, idxs in domain_to_idxs.items():
            np.save(ds_path / f"{domain.replace(' ', '_')}_idxs.npy", np.asarray(idxs))

    for domain in filter_domains:
        domain_idxs_path = ds_path / f"{domain.replace(' ', '_')}_idxs.npy"
        domain_idxs = np.load(domain_idxs_path)
        domain_to_idxs[domain] = domain_idxs

    return domain_to_idxs


def resample(ds_dir, cache_ds_dir, num_to_retrieve):
    if args.pack_every_2_examples:
        suffix = '_pack'
    else:
        suffix = '_nopack'
    retrieved_dir = ds_dir / 'retrieved'
    retrieved_path_cache = cache_ds_dir / f'retrieved_{num_to_retrieve}{suffix}.jsonl'

    retrieved_dir.mkdir(parents=True, exist_ok=True)
    retrieved_path = retrieved_dir / f'retrieved_{num_to_retrieve}{suffix}.jsonl'

    total_lines = dsname_to_args['pile']['total_lines']

    if args.qualityfilter:
        quality_scores = np.load(dsname_to_args['pile']['quality_scores'])
        global_mask = get_quality_mask(quality_scores)

    if args.ds_name == 'wiki_and_books':
        # compute the wikipedia and book masks and filter out
        filter_domains = ['Wikipedia (en)', 'BookCorpus2', 'Books3', 'Gutenberg (PG-19)']
        domain_to_idxs = compute_domain_idxs(filter_domains)
        for domain, idxs in domain_to_idxs.items():
            # ignore wiki and books during retrieval
            mask = np.ones(total_lines).astype(bool)
            mask[idxs] = False
            global_mask = global_mask & mask

    # merge logratio chunks
    logratios_file = retrieved_dir / 'logratios.npy'
    chunk_dir = Path(ds_dir) / f'logratio_chunks_ngrams{args.ngrams}_buckets{args.num_buckets}'
    if not logratios_file.exists():
        logratios = []
        for i in subsets:
            curr_logratios_file = chunk_dir / f'{i}.npy'
            logratios.append(np.load(curr_logratios_file))
        logratios = np.concatenate(logratios)
        np.save(logratios_file, logratios)
    else:
        logratios = np.load(logratios_file)

    assert(len(logratios) == total_lines)

    # noise the logratios
    logratios = logratios[global_mask]
    logratios += np.random.gumbel(size=len(logratios))

    nonzero_idxs = np.where(global_mask)[0]
    # choose top k
    chosen_idxs = np.argpartition(-logratios, num_to_retrieve)[:num_to_retrieve]
    chosen_idxs = nonzero_idxs[chosen_idxs]

    if args.ds_name == 'wiki_and_books':
        # add in some wikipedia and bookcorpus
        all_domain_idxs = []
        for domain, idxs in domain_to_idxs.items():
            # add 2 million from each domain (1million packed)
            if domain == 'Wikipedia (en)':
                num_to_add = 2000000
            else:
                num_to_add = 2000000 // 3
            np.random.shuffle(idxs)
            domain_chosen_idxs = idxs[:num_to_add]
            all_domain_idxs.append(domain_chosen_idxs)
        chosen_idxs = np.concatenate([chosen_idxs] + all_domain_idxs)

    global_mask = np.zeros(len(global_mask)).astype(bool)
    global_mask[chosen_idxs] = True

    del logratios
    del nonzero_idxs
    del chosen_idxs

    combined_streaming_ds = load_dataset(
            'json',
            data_files=dsname_to_args['pile']['task_name'],
            streaming=True)['train']

    prev_ex = None
    with open(retrieved_path_cache, 'w') as fout:
        for i, curr_ex in tqdm(enumerate(combined_streaming_ds), miniters=1000000, total=total_lines):
            if global_mask[i]:
                if args.pack_every_2_examples and prev_ex is not None:
                    prev_ex['contents'] += curr_ex['contents']
                    prev_ex['metadata']['pile_set_name'] = [
                        prev_ex['metadata']['pile_set_name'],
                        curr_ex['metadata']['pile_set_name']]
                    fout.write(json.dumps(prev_ex).strip() + '\n')
                    prev_ex = None
                elif args.pack_every_2_examples and prev_ex is None:
                    prev_ex = curr_ex
                else:
                    fout.write(json.dumps(curr_ex).strip() + '\n')

    shutil.move(retrieved_path_cache, retrieved_path)


def linecount(filename):
    out = subprocess.Popen(['wc', '-l', filename],
                           stdout=subprocess.PIPE).communicate()[0]
    return int(out.strip().partition(b' ')[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data selection with DSIR')
    parser.add_argument('--pile_path', type=str, help='path to pile')
    parser.add_argument('--ds_name', type=str, help='dataset name')
    parser.add_argument('--output_dir', type=str, help='output path')
    parser.add_argument('--num_to_retrieve', type=int, default=25600000, help='Number of examples to retrieve')
    parser.add_argument('--cache_dir', type=str,
                        help='cache directory for datasets')
    parser.add_argument('--num_proc', type=int, help='number of threads')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite importance weights')
    parser.add_argument('--overwrite_preprocess', action='store_true', help='overwrite data preprocessing')
    parser.add_argument('--ngrams', type=int, default=2, help='N in N-grams. 2 means both unigram and bigram.')
    parser.add_argument('--num_buckets', type=int, default=10000, help='number of ngram hash buckets')
    parser.add_argument('--pipeline_step', type=str, help='which step of pipeline to run. (imporance_weights, resample)')
    parser.add_argument('--chunk_idx', type=str, default='01', help='which chunk of prediction')
    parser.add_argument('--num_chunks', type=int, default=29, help='Number of chunks')
    parser.add_argument('--qualityfilter', action='store_true', help='whether to implement quality filtering')
    parser.add_argument('--pack_every_2_examples', action='store_true', help='whether to pack two examples together to get longer examples')
    args = parser.parse_args()
    random.seed(121)

    chunked_dir = 'chunked'
    dsname_to_args['pile_val'].update(
            {'task_name': [f'{args.pile_path}/{chunked_dir}/VAL_128/val_128.json'],
             'quality_scores': f'{args.pile_path}/{chunked_dir}/VAL_128/val_128.json_qualityscores.npz'}
             )

    #    XXX: if using fewer subsets of the Pile, please change the subsets list and the total_lines variable.
    #    We provde an example below (note that using linecount can take a long time for large numbers of subsets - we suggest running this once and hardcoding the number):
    # subsets = ['00']
    # total_lines = sum([linecount(f'{args.pile_path}/{chunked_dir}/{subset}_128/{subset}_128.json') for subset in subsets])
    # dsname_to_args['pile']['total_lines'] = total_lines

    dsname_to_args['pile'].update(
        {'task_name': [f'{args.pile_path}/{chunked_dir}/{subset}_128/{subset}_128.json' for subset in subsets],
         'quality_scores': f'{args.pile_path}/{chunked_dir}/combined_all.json_qualityscores.npz'}
            )

    cache_ds_dir = Path(args.cache_dir) / 'ngram_cache' / args.ds_name
    cache_ds_dir.mkdir(exist_ok=True, parents=True)
    ds_dir = Path(args.output_dir) / args.ds_name
    ds_dir.mkdir(exist_ok=True, parents=True)

    if args.ds_name == 'wiki_and_books':
        filter_domains = {'Wikipedia (en)', 'BookCorpus2'}
        ds_ngram_dist = compute_ngrams_pile(
                path=dsname_to_args['pile_val']['task_name'][0],
                ngrams=args.ngrams,
                num_buckets=args.num_buckets,
                filter_domains=filter_domains,
                )
        ds_ngram_dist = ds_ngram_dist / ds_ngram_dist.sum()
    else:
        ds_ngram_dist = compute_ngrams_hf(args.ds_name, ds_dir, cache_ds_dir, ngrams=args.ngrams, num_buckets=args.num_buckets)
        ds_ngram_dist = ds_ngram_dist / ds_ngram_dist.sum()

    pile_dist = compute_ngrams_pile(
            path=dsname_to_args['pile_val']['task_name'][0],
            ngrams=args.ngrams,
            num_buckets=args.num_buckets,
            )
    pile_dist = pile_dist / pile_dist.sum()

    if args.pipeline_step == 'importance_weights':
        _ = compute_importance_weights(
                path=f"{args.pile_path}/{chunked_dir}/{args.chunk_idx}_128/{args.chunk_idx}_128.json",
                ds_dir=ds_dir,
                chunk_idx=args.chunk_idx,
                target_dist=ds_ngram_dist,
                pile_dist=pile_dist,
                ngrams=args.ngrams,
                num_buckets=args.num_buckets,
                )
    elif args.pipeline_step == 'resample':
        resample(ds_dir, cache_ds_dir, args.num_to_retrieve)
