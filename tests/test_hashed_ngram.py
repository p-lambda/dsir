import pytest
from pathlib import Path
import numpy as np
import json
import shutil

from data_selection.hashed_ngram_dsir import HashedNgramDSIR, hash_buckets, get_ngram_counts


toy_dataset = Path(__file__).parent / "toy_pile_data.jsonl"
raw_datasets = [str(toy_dataset)] * 2
target_datasets = [str(Path(__file__).parent / "toy_target_data.jsonl")]
separated_target_datasets = [str(Path(__file__).parent / "toy_target_data.jsonl"), str(Path(__file__).parent / "toy_target_data_2.jsonl")]


def parse_example_fn(ex):
    return ex['contents']


@pytest.fixture
def dsir_obj():
    dsir = HashedNgramDSIR(
            raw_datasets=raw_datasets,
            target_datasets=target_datasets,
            cache_dir='/tmp/dsir_params',
            raw_parse_example_fn=parse_example_fn,
            target_parse_example_fn=parse_example_fn,
            num_proc=2,
            ngrams=2,
            num_buckets=10000)

    yield dsir

    if Path('/tmp/dsir_params').exists():
        shutil.rmtree('/tmp/dsir_params')


@pytest.fixture
def dsir_obj_diffparams():
    dsir = HashedNgramDSIR(
            raw_datasets=raw_datasets,
            target_datasets=target_datasets,
            cache_dir='/tmp/dsir_params',
            raw_parse_example_fn=parse_example_fn,
            target_parse_example_fn=parse_example_fn,
            num_proc=2,
            ngrams=3,
            num_buckets=50000)

    yield dsir

    if Path('/tmp/dsir_params').exists():
        shutil.rmtree('/tmp/dsir_params')


@pytest.fixture
def dsir_obj_septarget():
    dsir = HashedNgramDSIR(
            raw_datasets=raw_datasets,
            target_datasets=separated_target_datasets,
            cache_dir='/tmp/dsir_params',
            raw_parse_example_fn=parse_example_fn,
            target_parse_example_fn=parse_example_fn,
            num_proc=2,
            ngrams=2,
            num_buckets=10000,
            separate_targets=True,
            target_proportions=[0.5, 0.5])

    yield dsir

    if Path('/tmp/dsir_params').exists():
        shutil.rmtree('/tmp/dsir_params')


def test_hash_buckets():
    bucket = hash_buckets('alice')
    bucket_2 = hash_buckets('alice went')

    assert bucket == 6720
    assert bucket_2 == 114


def test_get_ngram_counts():
    line = 'Alice went to the store'

    counts = get_ngram_counts(line, n=2, num_buckets=10000)
    assert counts.shape == (10000,)
    assert counts.sum() == 9

    bucket = hash_buckets('alice')
    bucket_2 = hash_buckets('alice went')

    assert counts[bucket] > 0
    assert counts[bucket_2] > 0


def test_virtual_shards(dsir_obj):
    assert len(dsir_obj._get_virtually_sharded_datasets(raw_datasets)) == 2


def test_length_metadata(dsir_obj):
    text = "Alice walked into the store\n\n()()($#%@?)@(#(*"
    length = len(dsir_obj.tokenizer(text))
    feats = dsir_obj.featurizer(text)
    assert length == dsir_obj.get_perexample_metadata(ex=None, features=feats)


def test_fit(dsir_obj):
    dsir_obj.fit_importance_estimator(num_tokens_to_fit='all')

    assert dsir_obj.raw_probs is not None
    assert dsir_obj.raw_probs.shape == (10000,)
    assert dsir_obj.raw_probs.sum() == 1.0
    assert dsir_obj.target_probs is not None
    assert dsir_obj.target_probs.shape == (10000,)
    assert dsir_obj.target_probs.sum() == 1.0
    assert dsir_obj.log_diff is not None
    assert dsir_obj.log_diff.shape == (10000,)
    assert np.allclose(dsir_obj.log_diff, np.log(dsir_obj.target_probs + 1e-8) - np.log(dsir_obj.raw_probs + 1e-8))


    dsir_obj.fit_importance_estimator(num_tokens_to_fit='auto')

    assert dsir_obj.raw_probs is not None
    assert dsir_obj.raw_probs.shape == (10000,)
    assert dsir_obj.raw_probs.sum() == 1.0
    assert dsir_obj.target_probs is not None
    assert dsir_obj.target_probs.shape == (10000,)
    assert dsir_obj.target_probs.sum() == 1.0
    assert dsir_obj.log_diff is not None
    assert dsir_obj.log_diff.shape == (10000,)
    assert np.allclose(dsir_obj.log_diff, np.log(dsir_obj.target_probs + 1e-8) - np.log(dsir_obj.raw_probs + 1e-8))


    dsir_obj.fit_importance_estimator(num_tokens_to_fit=100000)

    assert dsir_obj.raw_probs is not None
    assert dsir_obj.raw_probs.shape == (10000,)
    assert dsir_obj.raw_probs.sum() == 1.0
    assert dsir_obj.target_probs is not None
    assert dsir_obj.target_probs.shape == (10000,)
    assert dsir_obj.target_probs.sum() == 1.0
    assert dsir_obj.log_diff is not None
    assert dsir_obj.log_diff.shape == (10000,)
    assert np.allclose(dsir_obj.log_diff, np.log(dsir_obj.target_probs + 1e-8) - np.log(dsir_obj.raw_probs + 1e-8))



def test_compute(dsir_obj):
    dsir_obj.fit_importance_estimator()

    dsir_obj.compute_importance_weights()

    log_importance_weights = [
        np.load(dsir_obj.log_importance_weights_dir / f"{i}.npy") for i in range(2)]
    assert len(log_importance_weights) == len(raw_datasets)
    assert log_importance_weights[0].shape == (1000,)

def test_resample(dsir_obj):
    dsir_obj.fit_importance_estimator()

    dsir_obj.compute_importance_weights()

    dsir_obj.resample(out_dir='/tmp/resampled', num_to_sample=2, cache_dir='/tmp/resampled_cache')

    assert Path('/tmp/resampled').exists()
    assert not Path('/tmp/resampled_cache').exists()

    all_lines = []
    for i in range(dsir_obj.num_proc):
        with open(f'/tmp/resampled/{i}.jsonl', 'r') as f:
            lines = f.readlines()
            all_lines += lines

    assert len(all_lines) == 2

    for line in all_lines:
        ex = json.loads(line)
        assert ex['id'] == 0
        length = len(dsir_obj.tokenizer(dsir_obj.raw_parse_example_fn(ex)))
        assert length >= dsir_obj.min_example_length

    shutil.rmtree('/tmp/resampled')
    if Path('/tmp/resampled_cache').exists():
        shutil.rmtree('/tmp/resampled_cache')

def test_resample_diffparams(dsir_obj_diffparams):
    dsir_obj = dsir_obj_diffparams
    dsir_obj.fit_importance_estimator()

    dsir_obj.compute_importance_weights()

    dsir_obj.resample(out_dir='/tmp/resampled', num_to_sample=2, cache_dir='/tmp/resampled_cache')

    assert Path('/tmp/resampled').exists()
    assert not Path('/tmp/resampled_cache').exists()

    all_lines = []
    for i in range(dsir_obj.num_proc):
        with open(f'/tmp/resampled/{i}.jsonl', 'r') as f:
            lines = f.readlines()
            all_lines += lines

    assert len(all_lines) == 2

    for line in all_lines:
        ex = json.loads(line)
        assert ex['id'] == 0
        length = len(dsir_obj.tokenizer(dsir_obj.raw_parse_example_fn(ex)))
        assert length >= dsir_obj.min_example_length

    shutil.rmtree('/tmp/resampled')
    if Path('/tmp/resampled_cache').exists():
        shutil.rmtree('/tmp/resampled_cache')


def test_resample_septarget(dsir_obj_septarget):
    dsir_obj = dsir_obj_septarget
    dsir_obj.fit_importance_estimator()

    dsir_obj.compute_importance_weights()

    dsir_obj.resample(out_dir='/tmp/resampled', num_to_sample=2, cache_dir='/tmp/resampled_cache')

    assert Path('/tmp/resampled').exists()
    assert not Path('/tmp/resampled_cache').exists()
    assert np.allclose(dsir_obj.target_proportions - 0.5, 0.0)

    all_lines = []
    for i in range(dsir_obj.num_proc):
        with open(f'/tmp/resampled/{i}.jsonl', 'r') as f:
            lines = f.readlines()
            all_lines += lines

    assert len(all_lines) == 2

    all_ids = []
    for line in all_lines:
        ex = json.loads(line)
        all_ids.append(ex['id'])
        length = len(dsir_obj.tokenizer(dsir_obj.raw_parse_example_fn(ex)))
        assert length >= dsir_obj.min_example_length

    assert len(set(all_ids)) == 2
    assert min(all_ids) == 0
    assert max(all_ids) == 2

    shutil.rmtree('/tmp/resampled')
    if Path('/tmp/resampled_cache').exists():
        shutil.rmtree('/tmp/resampled_cache')


def test_resample_virtual_sharding():
    dsir_obj = HashedNgramDSIR(
            raw_datasets=raw_datasets,
            target_datasets=target_datasets,
            cache_dir='/tmp/dsir_params',
            raw_parse_example_fn=parse_example_fn,
            target_parse_example_fn=parse_example_fn,
            num_proc=15,
            ngrams=2,
            num_buckets=10000)

    assert len(dsir_obj._get_virtually_sharded_datasets(raw_datasets)) == 15

    dsir_obj.fit_importance_estimator()

    dsir_obj.compute_importance_weights()

    dsir_obj.resample(out_dir='/tmp/resampled_virtual', num_to_sample=2, cache_dir='/tmp/resampled_cache_virtual')

    assert Path('/tmp/resampled_virtual').exists()
    assert not Path('/tmp/resampled_cache_virtual').exists()

    all_lines = []
    for i in range(15):
        with open(f'/tmp/resampled_virtual/{i}.jsonl', 'r') as f:
            lines = f.readlines()
            all_lines += lines

    assert len(all_lines) == 2

    for line in all_lines:
        ex = json.loads(line)
        assert ex['id'] == 0
        length = len(dsir_obj.tokenizer(dsir_obj.raw_parse_example_fn(ex)))
        assert length >= dsir_obj.min_example_length

    shutil.rmtree('/tmp/resampled_virtual')
    if Path('/tmp/resampled_virtual').exists():
        shutil.rmtree('/tmp/resampled_virtual')


def test_smoothing(dsir_obj):
    dsir_obj.fit_importance_estimator()

    target_probs_1 = dsir_obj.target_probs

    smoothing_param = 1.0
    dsir_obj_2 = HashedNgramDSIR(
            raw_datasets=raw_datasets,
            target_datasets=target_datasets,
            cache_dir='/tmp/dsir_params_2',
            raw_parse_example_fn=parse_example_fn,
            target_parse_example_fn=parse_example_fn,
            num_proc=2,
            ngrams=2,
            num_buckets=10000,
            target_laplace_smoothing=smoothing_param)
    dsir_obj_2.fit_importance_estimator()

    target_probs_2 = dsir_obj_2.target_probs

    total_counts = None
    with open(target_datasets[0], 'r') as f:
        for line in f:
            ex = json.loads(line)
            text = parse_example_fn(ex)
            counts = get_ngram_counts(text, n=2, num_buckets=10000)
            if total_counts is None:
                total_counts = counts
            else:
                total_counts += counts

    total_tokens = total_counts.sum()

    assert np.allclose(total_counts / total_tokens, target_probs_1)

    assert np.allclose((target_probs_1 * total_tokens + smoothing_param) / (total_tokens + smoothing_param * len(target_probs_1)), target_probs_2)

    if Path('/tmp/dsir_params').exists():
        shutil.rmtree('/tmp/dsir_params_2')


def test_save_load(dsir_obj):
    dsir_obj.fit_importance_estimator()

    dsir_obj.compute_importance_weights()

    dsir_obj.save('/tmp/dsir.pkl')

    dsir_obj_2 = HashedNgramDSIR([], [], '/tmp/cache')
    dsir_obj_2.load('/tmp/dsir.pkl', exclude_keys=['raw_datasets', 'target_datasets', 'cache_dir'])
    assert np.allclose(dsir_obj_2.raw_probs, dsir_obj.raw_probs)
    assert np.allclose(dsir_obj_2.target_probs, dsir_obj.target_probs)
    assert np.allclose(dsir_obj_2.log_diff, dsir_obj.log_diff)
    assert dsir_obj_2.num_buckets == dsir_obj.num_buckets
    assert dsir_obj_2.ngrams == dsir_obj.ngrams

    dsir_obj_2 = HashedNgramDSIR([], [], '/tmp/cache')
    dsir_obj_2.load('/tmp/dsir.pkl')

    assert np.allclose(dsir_obj_2.raw_probs, dsir_obj.raw_probs)
    assert np.allclose(dsir_obj_2.target_probs, dsir_obj.target_probs)
    assert np.allclose(dsir_obj_2.log_diff, dsir_obj.log_diff)
    assert dsir_obj.cache_dir == dsir_obj_2.cache_dir
    assert dsir_obj_2.raw_datasets == dsir_obj.raw_datasets
    assert dsir_obj_2.target_datasets == dsir_obj.target_datasets
    assert dsir_obj_2.num_buckets == dsir_obj.num_buckets
    assert dsir_obj_2.ngrams == dsir_obj.ngrams
