from pathlib import Path
import shutil
from typing import List, Optional, Dict, Callable, Union
from collections.abc import Iterable
from json import dumps, loads
import hashlib
import pickle
from tqdm import tqdm
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize
from nltk import ngrams as get_ngrams
import numpy as np

from datasets import load_dataset

from dsir.base import DSIR
from dsir.utils import parallelize


wpt = WordPunctTokenizer()


def hash_buckets(text: str, num_buckets: int = 10000) -> int:
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % num_buckets


def get_ngram_counts(line: str,
                     n: int = 2,
                     num_buckets: int = 10000,
                     counts: np.ndarray = None,
                     tokenizer: Callable = word_tokenize) -> np.ndarray:
    '''
    If counts is given, modify counts.
    '''
    words = tokenizer(line.lower())

    if counts is None:
        counts = np.zeros(num_buckets, dtype=int)

    for w in words:
        counts[hash_buckets(w, num_buckets=num_buckets)] += 1
    for i in range(2, n + 1):
        for ng in list(get_ngrams(words, i)):
            ng = ' '.join(ng)
            counts[hash_buckets(ng, num_buckets=num_buckets)] += 1
    return counts


def default_parse_example_fn(ex: Dict) -> str:
    return ex['text']


class HashedNgramDSIR(DSIR):

    def __init__(self,
                 raw_datasets: List[str],
                 parse_example_fn: Callable[[Dict], str] = default_parse_example_fn,
                 num_proc: Optional[int] = None,
                 ngrams: int = 2,
                 num_buckets: int = 10000,
                 tokenizer: str = 'word_tokenize') -> None:
        '''Initialize the HashedNgramDSIR object.

        Args:
            raw_datasets: List of paths to jsonl files
            parse_example_fn: Function that takes in an example dict and returns a string.
                              Defaults to returning the 'text' field of the example.
            num_proc: number of processes to use for parallelization. Defaults to number of cores.
            ngrams: N in N-grams. 2 means both unigram and bigrams.
            num_buckets: number of buckets to hash ngrams into.
            tokenizer: word_tokenize or wordpunct
        '''
        super().__init__(
                raw_datasets=raw_datasets,
                parse_example_fn=parse_example_fn,
                num_proc=num_proc)
        if tokenizer == 'word_tokenize':
            self.tokenizer = word_tokenize
        elif tokenizer == 'wordpunct':
            self.tokenizer = wpt.tokenize
        else:
            raise ValueError('tokenizer not recognized')
        self.ngrams = ngrams
        self.num_buckets = num_buckets
        self.raw_probs = None
        self.target_probs = None
        self.log_diff = None
        self.log_importance_weights = None
        self.target_datasets = None

    def importance_estimator(self, text: str) -> float:
        ngram_feats = get_ngram_counts(text, tokenizer=self.tokenizer)
        return np.inner(ngram_feats, self.log_diff)

    def _fit_bow(self, paths: List[str], num_tokens_to_fit: Optional[int] = None) -> np.ndarray:

        def job(path: str):
            counts = np.zeros(self.num_buckets).astype(int)
            dataset = load_dataset(
                    'json',
                    data_files=[path],
                    streaming=True)['train']
            for ex in tqdm(dataset, miniters=10000, maxinterval=1000000):
                if self.parse_example_fn is not None:
                    text = self.parse_example_fn(ex)
                else:
                    text = ex
                counts = get_ngram_counts(text,
                                          n=self.ngrams,
                                          num_buckets=self.num_buckets,
                                          counts=counts,
                                          tokenizer=self.tokenizer)

                if num_tokens_to_fit is not None and counts.sum() > num_tokens_to_fit // len(paths):
                    break

            return counts

        all_counts = parallelize(job, paths, self.num_proc)
        counts = sum(all_counts)

        counts = counts / counts.sum()
        return counts

    def fit_importance_estimator(self, target_datasets: List[str], num_tokens_to_fit: Union[str, int] = 'all') -> None:
        '''Fit the importance estimator.
        Args:
            target_datasets: List of paths to jsonl files or HuggingFace datasets
            num_tokens_to_fit: number of tokens to fit the importance estimator on.
                               Set to "all" to fit on all tokens, and "auto" to determine
                               the number of tokens to fit on automatically (100k * num_buckets).
                               Set to an integer to fit on that many tokens.
        '''
        if num_tokens_to_fit == 'auto':
            num_tokens_to_fit = 100000 * self.num_buckets
        elif num_tokens_to_fit == 'all':
            num_tokens_to_fit = None

        self.raw_probs = self._fit_bow(self.raw_datasets, num_tokens_to_fit=num_tokens_to_fit)
        self.target_probs = self._fit_bow(target_datasets, num_tokens_to_fit=num_tokens_to_fit)

        self.log_diff = np.log(self.target_probs + 1e-8) - np.log(self.raw_probs + 1e-8)

    def compute_importance_weights(self) -> np.ndarray:
        def job(path: str):
            log_importance_weights = []

            dataset = load_dataset(
                    'json',
                    data_files=[path],
                    streaming=True)['train']

            for ex in tqdm(dataset, miniters=10000, maxinterval=1000000):
                if self.parse_example_fn is not None:
                    text = self.parse_example_fn(ex)
                else:
                    text = ex
                log_importance_weights.append(self.importance_estimator(text))
            log_importance_weights = np.asarray(log_importance_weights)
            return log_importance_weights
        self.log_importance_weights = parallelize(job, self.raw_datasets, self.num_proc)
        return self.log_importance_weights

    def save(self, path: str):
        super().save(path)

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.raw_probs is not None:
            np.save(str(path / 'raw_probs.npy'), self.raw_probs)
        if self.target_probs is not None:
            np.save(str(path / 'target_probs.npy'), self.target_probs)
        if self.log_diff is not None:
            np.save(str(path / 'log_diff.npy'), self.log_diff)

        metadata = {'num_buckets': self.num_buckets,
                    'ngrams': self.ngrams,
                    'raw_datasets': self.raw_datasets,
                    'target_datasets': self.target_datasets,
                    'parse_example_fn': self.parse_example_fn}

        # pickle the metadata
        with open(str(path / 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)


    def load(self, path: str):
        super().load(path)

        path = Path(path)
        raw_probs_path = path / 'raw_probs.npy'
        if raw_probs_path.exists():
            self.raw_probs = np.load(str(raw_probs_path))
        target_probs_path = path / 'target_probs.npy'
        if target_probs_path.exists():
            self.target_probs = np.load(str(target_probs_path))

        log_diff_path = path / 'log_diff.npy'
        if log_diff_path.exists():
            self.log_diff = np.load(str(log_diff_path))
            assert(
                np.allclose(self.log_diff, np.log(self.target_probs + 1e-8) - np.log(self.raw_probs + 1e-8)))

        metadata_path = path / 'metadata.pkl'
        with open(str(metadata_path), 'rb') as f:
            metadata = pickle.load(f)

        for k, v in metadata.items():
            setattr(self, k, v)
