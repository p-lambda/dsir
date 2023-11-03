# base DSIR class
import os
from typing import List, Optional, Dict, Callable, Iterable
import multiprocessing as mp
from pathlib import Path
import shutil
import pickle

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from data_selection.utils import parallelize


def default_load_dataset_fn(path: str) -> Iterable[Dict]:
    """Load dataset from path

    Args:
        path (str): path to dataset file
    """
    return load_dataset('json',
                        data_files=[path],
                        streaming=True)['train']


def default_parse_example_fn(ex: Dict) -> str:
    """Default parse function from example dict to string

    Args:
        ex (Dict): example dict
    """
    return ex['text']



def _iterate_virtually_sharded_dataset(dataset: Iterable, num_shards: int, shard_idx: int):
    for i, ex in enumerate(dataset):
        if i % num_shards == shard_idx:
            yield ex


class DSIR():
    """Base class for data selection with importance resampling (DSIR)."""

    def __init__(self,
                 raw_datasets: List[str],
                 target_datasets: List[str],
                 raw_load_dataset_fn: Callable[[str], Iterable[Dict]] = default_load_dataset_fn,
                 raw_parse_example_fn: Callable[[Dict], str] = default_parse_example_fn,
                 target_load_dataset_fn: Callable[[str], Iterable[Dict]] = default_load_dataset_fn,
                 target_parse_example_fn: Callable[[Dict], str] = default_parse_example_fn,
                 num_proc: Optional[int] = None):
        """
        Args:
            raw_datasets: List of data paths
            target_datasets: List of data paths
            raw_load_dataset_fn: Function to load raw dataset from path
            raw_parse_example_fn: a function that takes in an example dict and outputs a string
            target_load_dataset_fn: Function to load target dataset from path
            target_parse_example_fn: a function that takes in an example dict and outputs a string
            num_proc: num cpus to parallelize over. Parallelism is limited to the number of shards (len(raw_datasets))
        """
        self.raw_datasets = raw_datasets
        self.target_datasets = target_datasets
        self.raw_parse_example_fn = raw_parse_example_fn
        self.raw_load_dataset_fn = raw_load_dataset_fn
        self.target_parse_example_fn = target_parse_example_fn
        self.target_load_dataset_fn = target_load_dataset_fn
        if num_proc is None:
            try:
                # doesn't work on some systems
                self.num_proc = len(os.sched_getaffinity(0))
            except AttributeError:
                self.num_proc = mp.cpu_count()
        else:
            self.num_proc = num_proc
        self.log_importance_weights = None

    def _get_virtually_sharded_datasets(self, datasets: List[str]):
        """Return virtual shard parameters."""
        num_proc_per_shard = max(1, self.num_proc // len(datasets))
        if self.num_proc >= len(datasets):
            remainder = self.num_proc % len(datasets)
        else:
            remainder = 0

        shard_params = []
        for i, dataset in enumerate(datasets):
            curr_num_proc = num_proc_per_shard
            if i < remainder:
                curr_num_proc += 1
            for j in range(curr_num_proc):
                shard_params.append({'path': dataset, 'shard_idx': j, 'num_shards': curr_num_proc})
        return shard_params

    def importance_estimator(self, text: str) -> float:
        """Takes text and outputs an importance weight."""
        raise NotImplementedError

    def fit_importance_estimator(self) -> None:
        """Fits parameters needed to run self.importance_estimator.

        Args:
        """
        raise NotImplementedError

    def compute_importance_weights(self) -> np.ndarray:
        """Compute importance weights on raw dataset with self.importance_estimator. Returns importance weights and saves them in self.importance_weights."""
        raise NotImplementedError

    def resample(self, out_dir: str, num_to_sample: int, cache_dir: str = None, top_k: bool = False) -> None:
        """Resample raw dataset with self.importance_weights.

        Args:
            out_dir (str): path to save resampled dataset
            num_to_sample (int): number of samples to resample
            cache_dir (str): path to cache resampled dataset
            top_k (bool): if True, get top_k examples by importance weight instead of sampling
        """
        if cache_dir is None:
            cache_dir = out_dir

        out_dir = Path(out_dir)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        concat_log_importance_weights = np.concatenate(self.log_importance_weights)

        # noise the log_importance_weights
        if not top_k:
            concat_log_importance_weights += np.random.gumbel(size=len(concat_log_importance_weights))
        chosen_idxs = np.argpartition(-concat_log_importance_weights, num_to_sample)[:num_to_sample]
        global_mask = np.zeros(len(concat_log_importance_weights), dtype=bool)
        global_mask[chosen_idxs] = True

        del chosen_idxs
        del concat_log_importance_weights

        # split the global mask into per-dataset masks
        masks = []
        start_idx = 0
        for log_importance_weights in self.log_importance_weights:
            end_idx = start_idx + len(log_importance_weights)
            masks.append(global_mask[start_idx:end_idx])
            start_idx = end_idx

        def job(args: Dict):
            in_path = args['in_path']
            out_path = args['out_path']
            mask = args['mask']
            shard_idx = args['shard_idx']
            num_shards = args['num_shards']

            if Path(in_path).suffix == '.jsonl':
                # faster to not load json into dicts
                with open(out_path, 'w') as f:
                    with open(in_path, 'r') as f_in:
                        iterator = _iterate_virtually_sharded_dataset(f_in, num_shards, shard_idx)
                        for i, line in tqdm(enumerate(iterator), miniters=10000, maxinterval=1000000):
                            if mask[i]:
                                f.write(line.strip() + '\n')
            else:
                dataset = self.raw_load_dataset_fn(in_path)

                with open(out_path, 'w') as f:
                    iterator = _iterate_virtually_sharded_dataset(dataset, num_shards, shard_idx)
                    for i, ex in tqdm(enumerate(iterator), miniters=10000, maxinterval=1000000):
                        if mask[i]:
                            f.write(dumps(ex) + '\n')
                del dataset

        sharded_raw_datasets = self._get_virtually_sharded_datasets(self.raw_datasets)
        args = [{'out_path': cache_dir / f"{i}.jsonl",
                 'in_path': shard_params['path'],
                 'mask': masks[i],
                 'shard_idx': shard_params['shard_idx'],
                 'num_shards': shard_params['num_shards']}
                for i, shard_params in enumerate(sharded_raw_datasets)]

        parallelize(job, args, self.num_proc)

        # move the cache_dir to out_dir
        shutil.move(str(cache_dir), str(out_dir))

    def save(self, path: str) -> None:
        """Save parameters to save computation"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.log_importance_weights is not None:
            np.save(str(path / 'log_importance_weights.npy'), self.log_importance_weights)

        metadata = {'raw_datasets': self.raw_datasets,
                    'target_datasets': self.target_datasets,
                    'raw_parse_example_fn': self.raw_parse_example_fn,
                    'raw_load_dataset_fn': self.raw_load_dataset_fn,
                    'target_parse_example_fn': self.target_parse_example_fn,
                    'target_load_dataset_fn': self.target_load_dataset_fn,
                    }
        # pickle the metadata
        with open(str(path / 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)

    def load(self, path: str) -> None:
        """Load saved parameters"""
        path = Path(path)
        log_importance_weights_path = path / 'log_importance_weights.npy'
        if log_importance_weights_path.exists():
            self.log_importance_weights = np.load(str(log_importance_weights_path))

        metadata_path = path / 'metadata.pkl'
        with open(str(metadata_path), 'rb') as f:
            metadata = pickle.load(f)

        for k, v in metadata.items():
            setattr(self, k, v)

