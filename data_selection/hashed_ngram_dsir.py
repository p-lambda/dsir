from pathlib import Path
import shutil
from typing import List, Optional, Dict, Callable, Union, Iterable
from json import dumps, loads
import hashlib
import pickle
from tqdm import tqdm
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import word_tokenize
from nltk import ngrams as get_ngrams
import numpy as np

from data_selection.base import (
        DSIR,
        default_load_dataset_fn,
        default_parse_example_fn,
        _iterate_virtually_sharded_dataset,
)

from data_selection.utils import parallelize


wpt = WordPunctTokenizer()


def hash_buckets(text: str, num_buckets: int = 10000) -> int:
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % num_buckets


def get_ngram_counts(line: str,
                     n: int = 2,
                     num_buckets: int = 10000,
                     counts: Optional[np.ndarray] = None,
                     tokenizer: Callable = wpt.tokenize) -> np.ndarray:
    '''Return ngram count features given a string.

    Args:
        line: string to get ngram counts from
        n: n in ngrams
        num_buckets: number of buckets to hash ngrams into
        counts: pre-initialized counts array
        tokenizer: tokenization function to use. Defaults to word_tokenize from nltk
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


class HashedNgramDSIR(DSIR):

    def __init__(self,
                 raw_datasets: List[str],
                 target_datasets: List[str],
                 cache_dir: str,
                 raw_load_dataset_fn: Callable[[str], Iterable[Dict]] = default_load_dataset_fn,
                 raw_parse_example_fn: Callable[[Dict], str] = default_parse_example_fn,
                 target_load_dataset_fn: Callable[[str], Iterable[Dict]] = default_load_dataset_fn,
                 target_parse_example_fn: Callable[[Dict], str] = default_parse_example_fn,
                 num_proc: Optional[int] = None,
                 ngrams: int = 2,
                 num_buckets: int = 10000,
                 tokenizer: str = 'wordpunct',
                 min_example_length: int = 100) -> None:
        '''Initialize the HashedNgramDSIR object.

        Args:
            raw_datasets: List of data paths
            target_datasets: List of data paths
            cache_dir: place to store cached log_importance_weights
            load_dataset_fn: Function to load a dataset from a path. Defaults to default_load_dataset_fn.
            parse_example_fn: Function that takes in an example dict and returns a string.
                              Defaults to returning the 'text' field of the example.
            num_proc: number of processes to use for parallelization. Defaults to number of cores.
            ngrams: N in N-grams. 2 means both unigram and bigrams.
            num_buckets: number of buckets to hash ngrams into.
            tokenizer: word_tokenize or wordpunct
            min_example_length: minimum number of tokens in an example to be considered.
        '''
        super().__init__(
                raw_datasets=raw_datasets,
                target_datasets=target_datasets,
                cache_dir=cache_dir,
                raw_load_dataset_fn=raw_load_dataset_fn,
                raw_parse_example_fn=raw_parse_example_fn,
                target_load_dataset_fn=target_load_dataset_fn,
                target_parse_example_fn=target_parse_example_fn,
                num_proc=num_proc)
        if tokenizer == 'word_tokenize':
            self.tokenizer = word_tokenize
        elif tokenizer == 'wordpunct':
            self.tokenizer = wpt.tokenize
        else:
            raise ValueError('tokenizer not recognized')
        self.ngrams = ngrams
        self.num_buckets = num_buckets
        self.min_example_length = min_example_length
        self.raw_probs = None
        self.target_probs = None
        self.log_diff = None

    def featurizer(self, text: str) -> np.ndarray:
        return get_ngram_counts(text, tokenizer=self.tokenizer)

    def importance_estimator(self, features: np.ndarray) -> float:
        return np.inner(features, self.log_diff)

    def get_perexample_metadata(self, ex: Dict, features: np.ndarray) -> int:
        """Returns the example length."""
        remainder = self.ngrams * (self.ngrams - 1) / 2
        return (features.sum() + remainder) // self.ngrams

    def perexample_metadata_filter(self, concat_metadata: np.ndarray) -> np.array:
        """Filters out short examples."""
        return concat_metadata >= self.min_example_length

    def _fit_bow(self,
                 paths: List[str],
                 num_tokens_to_fit: Optional[int] = None,
                 load_dataset_fn: Callable[[str], Iterable[Dict]] = default_load_dataset_fn,
                 parse_example_fn: Callable[[Dict], str] = default_parse_example_fn) -> np.ndarray:

        sharded_datasets = self._get_virtually_sharded_datasets(paths)
        def job(args: Dict):
            path = args['path']
            num_shards = args['num_shards']
            shard_idx = args['shard_idx']

            counts = np.zeros(self.num_buckets).astype(int)
            dataset = load_dataset_fn(path)
            iterator = _iterate_virtually_sharded_dataset(dataset, num_shards, shard_idx)
            for ex in tqdm(iterator, miniters=10000, maxinterval=1000000):
                if parse_example_fn is not None:
                    text = parse_example_fn(ex)
                else:
                    text = ex
                counts = get_ngram_counts(text,
                                          n=self.ngrams,
                                          num_buckets=self.num_buckets,
                                          counts=counts,
                                          tokenizer=self.tokenizer)

                if num_tokens_to_fit is not None and counts.sum() > num_tokens_to_fit // len(sharded_datasets):
                    break

            return counts

        all_counts = parallelize(job, sharded_datasets, self.num_proc)
        counts = sum(all_counts)

        counts = counts / counts.sum()
        return counts

    def fit_importance_estimator(self, num_tokens_to_fit: Union[str, int] = 'auto') -> None:
        '''Fit the importance estimator.
        Args:
            target_datasets: List of paths to jsonl-like files loadable by HuggingFace's load_dataset function.
            num_tokens_to_fit: number of tokens to fit the raw dataset importance estimator on.
                               Set to "all" to fit on all tokens, and "auto" to determine
                               the number of tokens to fit on automatically (100k * num_buckets).
                               Set to an integer to fit on that many tokens.
        '''
        if num_tokens_to_fit == 'auto':
            num_tokens_to_fit = 100000 * self.num_buckets
        elif num_tokens_to_fit == 'all':
            num_tokens_to_fit = None

        self.raw_probs = self._fit_bow(
                self.raw_datasets,
                num_tokens_to_fit=num_tokens_to_fit,
                parse_example_fn=self.raw_parse_example_fn,
                load_dataset_fn=self.raw_load_dataset_fn)
        self.target_probs = self._fit_bow(
                self.target_datasets,
                num_tokens_to_fit=None,  # fit on all tokens for target
                parse_example_fn=self.target_parse_example_fn,
                load_dataset_fn=self.target_load_dataset_fn)

        self.log_diff = np.log(self.target_probs + 1e-8) - np.log(self.raw_probs + 1e-8)

    def save(self, path: str):
        super().save(path)

        path = Path(path)
        if self.raw_probs is not None:
            np.save(str(path / 'raw_probs.npy'), self.raw_probs)
        if self.target_probs is not None:
            np.save(str(path / 'target_probs.npy'), self.target_probs)
        if self.log_diff is not None:
            np.save(str(path / 'log_diff.npy'), self.log_diff)

        with open(str(path / 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)

        metadata.update({
            'num_buckets': self.num_buckets,
            'ngrams': self.ngrams,
            'tokenizer': self.tokenizer})

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

