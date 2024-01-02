# Usage
In general, DSIR aims to select data from the raw dataset that matches the feature distribution of the target data. Thus, the choice of feature space and importance estimator on this feature space can change the behavior of DSIR for different use-cases. Extending the base DSIR class in `base.py` is simple - follow the example in `hashed_ngram_dsir.py`.

### class data_selection.DSIR
Base class for DSIR.

- `raw_datasets`: List of data paths
- `target_datasets`: List of data paths
- `cache_dir`: Directory to store cached intermediates (log importance weights)
- `raw_load_dataset_fn`: Function to load raw dataset from path
- `raw_parse_example_fn`: a function that takes in an example dict and outputs a string
- `target_load_dataset_fn`: Function to load target dataset from path
- `target_parse_example_fn`: a function that takes in an example dict and outputs a string
- `num_proc`: num cpus to parallelize over. If None, use all available cpus.
- `separate_targets`: whether to select data separately for each target and then join them. For example, when including two target datasets, one natural language dataset and one code, the most heavily upweighted data when `separate_targets=False` may skew towards documents with a mix of natural language and code, such as StackExchange. When `separate_targets=True`, two separate DSIR runs will occur in parallel, selecting a mixture of documents using each target
- `target_proportions`: weighting across multiple targets if separate_targets=True. The proportions are on the document level. Set to None to weight by the size (in tokens) of each target dataset.

#### compute_importance_weights(self) -> None:
Compute importance weights on raw dataset with self.importance_estimator.
Saves importance weights in self.log_importance_weights_dir / {index}.npy in chunks indexed by index.
Also saves other per-example metadata (numpy arrays) in self.perexample_metadata_dir / {index}.npy."""

#### resample(self, out_dir: str, num_to_sample: int, cache_dir: str = None, top_k: bool = False) -> None:
Resample raw dataset according to importance weights.

- `out_dir`: path to save resampled dataset
- `num_to_sample`: number of samples to resample
- `cache_dir`: path to cache resampled dataset
- `top_k`: if True, get top_k examples by importance weight instead of sampling


### class data_selection.HashedNgramDSIR
The main subclass we provide is DSIR with hashed n-gram features. This choice of feature space allows for efficient data selection over large datasets. 

- `raw_datasets`: List of data paths
- `target_datasets`: List of data paths
- `cache_dir`: place to store cached log_importance_weights
- `load_dataset_fn`: Function to load a dataset from a path. Defaults to default_load_dataset_fn.
- `parse_example_fn`: Function that takes in an example dict and returns a string. Defaults to returning the "text" field of the example.
- `num_proc`: number of processes to use for parallelization. Defaults to number of cores.
- `ngrams`: N in N-grams. 2 means both unigram and bigrams.
- `num_buckets`: number of buckets to hash ngrams into.
- `tokenizer`: word_tokenize or wordpunct
- `min_example_length`: minimum number of tokens in an example to be considered.
- `target_laplace_smoothing`: Smooth the target hash ngram distribution with this Laplace smoothing parameter, which is a pseudo-count. This could be useful for small target datasets.
- `separate_targets`: whether to select data separately for each target and then join them. For example, when including two target datasets, one natural language dataset and one code, the most heavily upweighted data when `separate_targets=False` may skew towards documents with a mix of natural language and code, such as StackExchange. When `separate_targets=True`, two separate DSIR runs will occur in parallel, selecting a mixture of documents using each target according to `target_proportions`.
- `target_proportions`: weighting across multiple targets if separate_targets=True. The proportions are on the document level. Set to None to weight by the size in tokens of each target dataset

#### fit_importance_estimator(self, num_tokens_to_fit: Union[str, int] = 'auto') -> None:
Fit the importance estimator.

- `num_tokens_to_fit`: number of tokens to fit the raw dataset importance estimator on. Set to "all" to fit on all tokens, and "auto" to determine the number of tokens to fit on automatically (100k * num_buckets). Set to an integer to fit on that many tokens.
