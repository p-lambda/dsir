# base DSIR class
import os
from collections.abc import Iterable
from typing import List, Optional, Dict, Callable
import multiprocessing as mp
from pathlib import Path
import shutil

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from dsir.utils import parallelize



class DSIR():
    '''Base class for data selection with importance resampling.'''

    def __init__(self, raw_datasets: List[str], parse_example_fn: Callable[[Dict], str] = None,  num_proc: Optional[int] = None):
        '''
        Args:
            raw_datasets: List of dataset paths to jsonl files
            parse_example_fn: a function that takes in an element from raw_datasets and outputs a string
            num_proc: num cpus to parallelize over. Parallelism is limited to the number of shards (len(raw_datasets))
        '''
        self.raw_datasets = raw_datasets
        self.parse_example_fn = parse_example_fn
        if num_proc is None:
            try:
                # doesn't work on some systems
                self.num_proc = len(os.sched_getaffinity(0))
            except AttributeError:
                self.num_proc = mp.cpu_count()
        else:
            self.num_proc = num_proc
        self.num_proc = min(self.num_proc, len(self.raw_datasets))
        self.log_importance_weights = None

    def importance_estimator(self, text: str) -> float:
        '''Takes text and outputs an importance weight.'''
        raise NotImplementedError

    def fit_importance_estimator(self, target_datasets: List[str]) -> None:
        '''Fits parameters needed to run self.importance_estimator.

        Args:
            target_datasets: List of dataset paths to jsonl files
        '''
        raise NotImplementedError

    def compute_importance_weights(self) -> np.ndarray:
        '''Compute importance weights on raw dataset with self.importance_estimator. Returns importance weights and saves them in self.importance_weights.'''
        raise NotImplementedError

    def resample(self, out_dir: str, num_to_sample: int, cache_dir: str = None, top_k: bool = False) -> None:
        '''Resample raw dataset with self.importance_weights.

        Args:
            out_dir (str): path to save resampled dataset
            num_to_sample (int): number of samples to resample
            cache_dir (str): path to cache resampled dataset
            top_k (bool): if True, get top_k examples by importance weight instead of sampling
        '''
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

            if Path(in_path).suffix == '.jsonl':
                # faster to not load json into dicts
                with open(out_path, 'w') as f:
                    with open(in_path, 'r') as f_in:
                        for i, line in tqdm(enumerate(f_in), miniters=10000, maxinterval=1000000):
                            if mask[i]:
                                f.write(line.strip() + '\n')
            else:
                dataset = load_dataset(
                        'json',
                        data_files=[in_path],
                        streaming=True)['train']

                with open(out_path, 'w') as f:
                    for i, ex in tqdm(enumerate(dataset), miniters=10000, maxinterval=1000000):
                        if mask[i]:
                            f.write(dumps(ex) + '\n')

        args = [{'out_path': cache_dir / f"{i}.jsonl", 'in_path': self.raw_datasets[i], 'mask': masks[i]}
                for i in range(len(self.raw_datasets))]

        parallelize(job, args, self.num_proc)

        # move the cache_dir to out_dir
        shutil.move(str(cache_dir), str(out_dir))

    def save(self, path: str):
        '''Save parameters to save computation'''
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.log_importance_weights is not None:
            np.save(str(path / 'log_importance_weights.npy'), self.log_importance_weights)

    def load(self, path: str):
        '''Load saved parameters'''
        path = Path(path)
        log_importance_weights_path = path / 'log_importance_weights.npy'
        if log_importance_weights_path.exists():
            self.log_importance_weights = np.load(str(log_importance_weights_path))
