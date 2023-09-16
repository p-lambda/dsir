from pathlib import Path
import os
import random
import argparse
import json
import shutil
from multiprocessing import Pool
from collections import defaultdict

from tqdm import tqdm
from nltk import word_tokenize
import numpy as np
from datasets import load_dataset
import fasttext

from dsir_pipeline import (
    dsname_to_args,
    subsets,
    get_quality_mask,
    linecount,

        )


def transform_text(text):
    return ' '.join(word_tokenize(text.lower()))


def batch_process(e, text_cols, label_col, fixed_label=None):
    sent = ' '.join([e[col] for col in text_cols])
    if fixed_label is not None:
        label = fixed_label
    else:
        label = e[label_col]
    sent = transform_text(sent)

    text = f'__label__{label} , {sent}'
    return {'text': text}


def reformat_dataset(ds_name, output_dir, cache_dir, num_proc=10, fixed_label=None, filter_domains=None):
    if args.qualityfilter and ds_name == 'pile':
        if filter_domains is not None:
            ds_output_dir = Path(output_dir) / (ds_name + '_qf_wikiandbooks')
        else:
            ds_output_dir = Path(output_dir) / (ds_name + '_qf')

        ds_output_dir.mkdir(exist_ok=True)

        quality_scores = np.load(dsname_to_args['pile']['quality_scores'])
        quality_mask = get_quality_mask(quality_scores)
    else:
        ds_output_dir = Path(output_dir) / ds_name
        ds_output_dir.mkdir(exist_ok=True)

    config = dsname_to_args[ds_name]
    text_cols, label_col = config["columns"], config["label"]
    print(text_cols, label_col)

    if ds_name != 'pile':
        ds = load_dataset(config["dataset_name"], config["task_name"],
                          cache_dir=cache_dir)
    else:
        ds = load_dataset(config["dataset_name"], data_files=config["task_name"],
                          cache_dir=cache_dir)

    for split in ds:
        print(split)
        if (ds_output_dir / f"{split}.txt").exists() and not args.overwrite_preprocess:
            continue
        split_ds = ds[split]

        column_names = list(iter(split_ds).__next__().keys())
        if ds_name == 'pile':
            column_names = [c for c in column_names if c != 'metadata']

        with open(f'{ds_output_dir}/{split}.txt', 'w') as f:
            for i, ex in tqdm(enumerate(split_ds)):
                if ds_name == 'pile':
                    if args.qualityfilter:
                        if not quality_mask[i]:
                            continue
                    domain = ex['metadata']['pile_set_name']
                    if filter_domains is not None and domain not in filter_domains:
                        continue

                sent = ' '.join([ex[col] for col in text_cols])
                if fixed_label is not None:
                    label = fixed_label
                else:
                    label = ex[label_col]
                sent = transform_text(sent)

                text = f'__label__{label} , {sent}'
                f.write(json.dumps({'text': text}) + '\n')

    return ds_output_dir


def replace_label(line, label):
    toks = line.split()
    new_label = '__label__' + str(label)

    rest = ' '.join(toks[2:])

    return f"{new_label} , {rest}"


def mix_dataset(ds_dir, pile_val_dir):
    """Interleave 2 datasets for fasttext classification: the target dataset ds_dir and the pile validation set"""
    ds_dir = Path(ds_dir)
    ds2_path = pile_val_dir / "train.txt"

    split = 'train'
    mixed_path_train = ds_dir / 'mixed-train.txt'
    mixed_path_val = ds_dir / 'mixed-val.txt'
    mixed_path_train_cache = cache_ds_dir / 'mixed-train.txt'
    mixed_path_val_cache = cache_ds_dir / 'mixed-val.txt'

    if args.qualityfilter:
        mixed_path_train = mixed_path_train.parent / f"{mixed_path_train.stem}-qf.txt"
        mixed_path_val = mixed_path_val.parent / f"{mixed_path_val.stem}-qf.txt"
        mixed_path_train_cache = mixed_path_train_cache.parent / f"{mixed_path_train_cache.stem}-qf.txt"
        mixed_path_val_cache = mixed_path_val_cache.parent / f"{mixed_path_val_cache.stem}-qf.txt"

    ds1_path = ds_dir / f"{split}.txt"

    if (mixed_path_train.exists() and mixed_path_val.exists()) and not args.overwrite_preprocess:
        return mixed_path_train, mixed_path_val

    num_lines1 = linecount(ds1_path)
    num_lines2 = linecount(ds2_path)

    num_lines = min(num_lines1, num_lines2)
    num_train = int(num_lines * 0.9)

    counter = 0
    with open(mixed_path_train_cache, 'w') as fout:
        with open(mixed_path_val_cache, 'w') as fout_val:
            with open(ds1_path, 'r') as f1:
                with open(ds2_path, 'r') as f2:
                    for line1, line2 in tqdm(zip(f1, f2), miniters=100000):
                        if counter < num_train:
                            if counter > 0:
                                fout.write('\n')
                            fout.write(replace_label(line1.strip(), label=1) + '\n')
                            fout.write(replace_label(line2.strip(), label=0))
                        else:
                            if counter - num_train > 0:
                                fout_val.write('\n')
                            fout_val.write(replace_label(line1.strip(), label=1) + '\n')
                            fout_val.write(replace_label(line2.strip(), label=0))
                        counter += 1
    shutil.move(mixed_path_train_cache, mixed_path_train)
    shutil.move(mixed_path_val_cache, mixed_path_val)
    return mixed_path_train, mixed_path_val


def prepare_fasttext_dataset(ds_dir):
    ds_dir = Path(ds_dir)
    split = 'train'
    ds1_path = ds_dir / f"{split}.txt"

    train_file = ds_dir / 'fasttext-train.txt'
    val_file = ds_dir / 'fasttext-val.txt'

    if (train_file.exists() and val_file.exists()) and not args.overwrite_preprocess:
        return train_file, val_file

    train_cache = cache_ds_dir / 'fasttext-train.txt'
    val_cache = cache_ds_dir / 'fasttext-val.txt'

    num_lines = linecount(ds1_path)
    num_train = int(num_lines * 0.9)

    counter = 0
    with open(train_cache, 'w') as fout:
        with open(val_cache, 'w') as fout_val:
            with open(ds1_path, 'r') as f:
                for line in tqdm(f, miniters=1000000):
                    if counter < num_train:
                        if counter > 0:
                            fout.write('\n')
                        fout.write(line.strip())
                    else:
                        if counter - num_train > 0:
                            fout_val.write('\n')
                        fout_val.write(line.strip())
                    counter += 1
    shutil.move(train_cache, train_file)
    shutil.move(val_cache, val_file)
    return train_file, val_file


def make_prediction(line, model):
    example = json.loads(line)
    transformed_example = transform_text(example['contents'])
    prediction = model.predict(transformed_example)

    label = int(prediction[0][0].split('__label__')[1])
    prob = np.amax(prediction[1])

    if label == 0:
        prob = 1 - prob

    return prob


model = None
def process(line):
    return make_prediction(line, model)


def make_prediction_chunk(ds_path, model_path, chunk_idx):
    global model
    model = fasttext.load_model(str(model_path))
    probs = []

    num_cpus = len(os.sched_getaffinity(0))
    pool = Pool(num_cpus)
    with open(ds_path, 'r') as f:
        for prob in pool.imap(process, f, chunksize=100000):
            probs.append(prob)

            if len(probs) % 100000 == 0:
                print(len(probs), flush=True)
    return np.asarray(probs)


def predict_chunk(model_path, ds_dir, chunk_idx):

    retrieved_dir = ds_dir / 'heuristic_cls_retrieved'
    retrieved_path_cache = cache_ds_dir / 'heuristic_cls_retrieved.jsonl'

    if args.word_vectors is not None:
        retrieved_dir = retrieved_dir.parent / f"{retrieved_dir.stem}_wordvecs"
        retrieved_path_cache = retrieved_path_cache.parent / f"{retrieved_path_cache.stem}_wordvecs.jsonl"

    if args.ngrams != 2:
        retrieved_dir = retrieved_dir.parent / f"{retrieved_dir.stem}_ngrams{args.ngrams}"
        retrieved_path_cache = retrieved_path_cache.parent / f"{retrieved_path_cache.stem}_ngrams{args.ngrams}.jsonl"

    if args.qualityfilter:
        retrieved_dir = retrieved_dir.parent / f"{retrieved_dir.stem}_qf"
        retrieved_path_cache = retrieved_path_cache.parent / f"{retrieved_path_cache.stem}_qf.jsonl"

    retrieved_dir.mkdir(parents=True, exist_ok=True)

    # run through dataset once to make the predictions
    ds_path = dsname_to_args['pile']['task_name']

    chunk_dir = retrieved_dir / 'chunks_bysubset'
    chunk_dir.mkdir(exist_ok=True)
    if args.word_vectors is not None:
        probabilities_file = chunk_dir / f'pred_probs_wordvecs_{chunk_idx}.npy'
    else:
        probabilities_file = chunk_dir / f'pred_probs_{chunk_idx}.npy'

    if not probabilities_file.exists() or args.overwrite:
        probabilities = make_prediction_chunk(ds_path, model_path, chunk_idx)
        np.save(probabilities_file, probabilities)


def compute_domain_idxs(filter_domains):
    ds_path = dsname_to_args['pile']['task_name']

    domain_to_idxs = defaultdict(list)
    todo_domains = []
    for domain in filter_domains:
        domain_idxs_path = Path(ds_path).parent / f"{domain.replace(' ', '_')}_idxs.npy"
        if not domain_idxs_path.exists():
            todo_domains.append(domain)

    combined_streaming_ds = load_dataset(
            'json',
            data_files=ds_path,
            streaming=True)['train']

    todo_domains = set(todo_domains)
    if len(todo_domains) > 0:
        for i, ex in tqdm(enumerate(combined_streaming_ds), miniters=1000000):
            domain = ex["metadata"]["pile_set_name"]
            if domain in todo_domains:
                domain_to_idxs[domain].append(i)
        for domain, idxs in domain_to_idxs.items():
            np.save(Path(ds_path).parent / f"{domain.replace(' ', '_')}_idxs.npy", np.asarray(idxs))

    for domain in filter_domains:
        domain_idxs_path = Path(ds_path).parent / f"{domain.replace(' ', '_')}_idxs.npy"
        domain_idxs = np.load(domain_idxs_path)
        domain_to_idxs[domain] = domain_idxs

    return domain_to_idxs


def retrieve_from_pile(model_path, num_to_retrieve, ds_dir):

    retrieved_dir = ds_dir / 'heuristic_cls_retrieved'
    retrieved_path_cache = cache_ds_dir / 'heuristic_cls_retrieved_{num_to_retrieve}.jsonl'

    if args.word_vectors is not None:
        retrieved_dir = retrieved_dir.parent / f"{retrieved_dir.stem}_wordvecs"
        retrieved_path_cache = retrieved_path_cache.parent / f"{retrieved_path_cache.stem}_wordvecs.jsonl"

    if args.ngrams != 2:
        retrieved_dir = retrieved_dir.parent / f"{retrieved_dir.stem}_ngrams{args.ngrams}"
        retrieved_path_cache = retrieved_path_cache.parent / f"{retrieved_path_cache.stem}_ngrams{args.ngrams}.jsonl"

    if args.qualityfilter:
        retrieved_dir = retrieved_dir.parent / f"{retrieved_dir.stem}_qf"
        retrieved_path_cache = retrieved_path_cache.parent / f"{retrieved_path_cache.stem}_qf.jsonl"

    retrieved_dir.mkdir(parents=True, exist_ok=True)
    retrieved_path = retrieved_dir / f'heuristic_cls_retrieved_{num_to_retrieve}.jsonl'

    retrieved_path = retrieved_path.parent / f"{retrieved_path.stem}_retrievemode{args.retrieval_mode}.jsonl"
    retrieved_path_cache = retrieved_path_cache.parent / f"{retrieved_path_cache.stem}_retrievemode{args.retrieval_mode}.jsonl"

    if args.pack_every_2_examples:
        retrieved_path = retrieved_path.parent / f"{retrieved_path.stem}_pack.jsonl"
        retrieved_path_cache = retrieved_path_cache.parent / f"{retrieved_path_cache.stem}_pack.jsonl"
    else:
        retrieved_path = retrieved_path.parent / f"{retrieved_path.stem}_nopack.jsonl"
        retrieved_path_cache = retrieved_path_cache.parent / f"{retrieved_path_cache.stem}_nopack.jsonl"

    ds_path = dsname_to_args['pile']['task_name']
    total_lines = dsname_to_args['pile']['total_lines']

    if args.word_vectors is not None:
        probabilities_file = retrieved_dir / 'pred_probs_wordvecs.npy'
    else:
        probabilities_file = retrieved_dir / 'pred_probs.npy'

    if not probabilities_file.exists() or args.overwrite:
        chunk_dir = retrieved_dir / 'chunks_bysubset'
        probabilities_ls = []
        for i in subsets:
            if args.word_vectors is not None:
                probabilities_file_chunk = chunk_dir / f'pred_probs_wordvecs_{i}.npy'
            else:
                probabilities_file_chunk = chunk_dir / f'pred_probs_{i}.npy'

            probabilities_ls.append(np.load(probabilities_file_chunk))
        probabilities = np.concatenate(probabilities_ls)

        np.save(probabilities_file, probabilities)
    else:
        probabilities = np.load(probabilities_file)

    assert(len(probabilities) == total_lines)

    if args.qualityfilter:
        quality_scores = np.load(dsname_to_args['pile']['quality_scores'])
        global_mask = get_quality_mask(quality_scores)
    else:
        global_mask = np.ones(total_lines).astype(bool)

    if args.ds_name == 'wiki_and_books':
        # compute the wikipedia and book masks and filter out
        filter_domains = ['Wikipedia (en)', 'BookCorpus2', 'Books3', 'Gutenberg (PG-19)']
        domain_to_idxs = compute_domain_idxs(filter_domains)
        for domain, idxs in domain_to_idxs.items():
            mask = np.ones(total_lines).astype(bool)
            mask[idxs] = False
            global_mask = global_mask & mask

    num_to_retrieve = min(num_to_retrieve, len(probabilities))

    def retrieve_mask(num, probabilities, mask=None):
        if mask is not None:
            mask = mask & global_mask
        else:
            mask = global_mask.copy()

        nonzero_idxs = np.where(mask)[0]

        if num <= mask.sum():
            if args.retrieval_mode == 'topk':
                chosen_idxs = np.argpartition(-probabilities[mask], num)[:num]
            elif args.retrieval_mode == 'pareto':
                pareto_rand_mask = np.zeros(len(nonzero_idxs)).astype(bool)
                masked_probs = probabilities[mask]
                while pareto_rand_mask.sum() < num:
                    rand = np.random.pareto(9, size=len(masked_probs))
                    pareto_rand_mask = pareto_rand_mask | (rand > (1 - masked_probs))
                    print("Pareto rand mask sum: ", pareto_rand_mask.sum())
                chosen_idxs = np.where(pareto_rand_mask)[0]
                np.random.shuffle(chosen_idxs)
                chosen_idxs = chosen_idxs[:num]

            else:
                raise ValueError("not implemented")

            chosen_idxs = nonzero_idxs[chosen_idxs]

            if args.ds_name == 'wiki_and_books':
                # add in some wikipedia and bookcorpus
                all_domain_idxs = []
                for domain, idxs in domain_to_idxs.items():
                    # add 2 million from each domain
                    if domain == 'Wikipedia (en)':
                        num_to_add = args.num_wiki_and_books // 2
                    else:
                        num_to_add = args.num_wiki_and_books // 6
                    np.random.shuffle(idxs)
                    domain_chosen_idxs = idxs[:num_to_add]
                    all_domain_idxs.append(domain_chosen_idxs)
                chosen_idxs = np.concatenate([chosen_idxs] + all_domain_idxs)

            new_mask = np.zeros(len(mask)).astype(bool)
            new_mask[chosen_idxs] = True
            mask = new_mask
        else:
            pass
        return mask

    mask = retrieve_mask(num_to_retrieve, probabilities)

    prev_line = None
    with open(retrieved_path_cache, 'w') as fout:
        with open(ds_path, 'r') as f:
            for i, line in tqdm(enumerate(f), total=total_lines, miniters=1000000):
                if mask[i]:
                    if args.pack_every_2_examples and prev_line is not None:
                        prev_ex = json.loads(prev_line)
                        curr_ex = json.loads(line)
                        prev_ex['contents'] += curr_ex['contents']
                        prev_ex['metadata']['pile_set_name'] = [
                            prev_ex['metadata']['pile_set_name'],
                            curr_ex['metadata']['pile_set_name']]
                        fout.write(json.dumps(prev_ex).strip() + '\n')
                        prev_line = None
                    elif args.pack_every_2_examples and prev_line is None:
                        prev_line = line
                    else:
                        example = json.loads(line)
                        example['score'] = probabilities[i]
                        fout.write(json.dumps(example).strip() + '\n')
    shutil.move(retrieved_path_cache, retrieved_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reformat datasets into fasttext and train classifier')
    parser.add_argument('--ds_name', type=str, help='dataset name')
    parser.add_argument('--output_dir', type=str, help='output path')
    parser.add_argument('--pile_path', type=str, help='path to pile')
    parser.add_argument('--num_to_retrieve', type=int, default=8192000,
                        help='amount of data to retrieve')
    parser.add_argument('--cache_dir', type=str,
                        help='cache directory for datasets')
    parser.add_argument('--word_vectors', default=None, type=str, help='path to word vectors. if None, word vectors are randomly initialized')
    parser.add_argument('--num_proc', type=int, default=2, help='number of threads')
    parser.add_argument('--overwrite', action='store_true', help='overwrite fasttext model and outputs')
    parser.add_argument('--overwrite_preprocess', action='store_true', help='overwrite data preprocessing')
    parser.add_argument('--ngrams', type=int, default=2, help='number of ngrams')
    parser.add_argument('--pipeline_step', type=str, help='which step of pipeline')
    parser.add_argument('--chunk_idx', type=str, default='01', help='which chunk of prediction')
    parser.add_argument('--retrieval_mode', default='pareto', type=str, help='type of retrieval')
    parser.add_argument('--qualityfilter', action='store_true', help='whether to implement quality filtering')
    parser.add_argument('--retrain_model', action='store_true', help='whether to retrain the model')
    parser.add_argument('--pack_every_2_examples', action='store_true', help='whether to pack')
    parser.add_argument('--num_wiki_and_books', type=int, default=4000000, help='number of random eaxmples from wikipedia and books')
    args = parser.parse_args()
    random.seed(121)

    chunked_dir = 'chunked'
    dsname_to_args['pile_val'].update(
            {'task_name': [f'{args.pile_path}/{chunked_dir}/VAL_128/val_128.json'],
             'quality_scores': f'{args.pile_path}/{chunked_dir}/VAL_128/val_128.json_qualityscores.npz'}
             )

    dsname_to_args['pile'].update(
        {'task_name': [f'{args.pile_path}/{chunked_dir}/{subset}_128/{subset}_128.json' for subset in subsets],
         'quality_scores': f'{args.pile_path}/{chunked_dir}/combined_all.json_qualityscores.npz'}
            )

    if args.ds_name == 'wiki_and_books':
        filter_domains = ['Wikipedia (en)', 'BookCorpus2']
        ds_dir = reformat_dataset('pile', args.output_dir, args.cache_dir,
                                  num_proc=args.num_proc, fixed_label="1",
                                  filter_domains=filter_domains)
    else:
        ds_dir = reformat_dataset(args.ds_name, args.output_dir, args.cache_dir,
                                  num_proc=args.num_proc)
    cache_ds_dir = Path(args.cache_dir) / 'retrieval_cache' / args.ds_name
    cache_ds_dir.mkdir(exist_ok=True, parents=True)

    pile_val_dir = reformat_dataset('pile', args.output_dir, args.cache_dir,
                                    num_proc=args.num_proc, fixed_label="0")

    train_file, val_file = mix_dataset(ds_dir, pile_val_dir)

    if args.word_vectors is not None:
        model_path = ds_dir / 'model_wordvecs.bin'
    else:
        model_path = ds_dir / 'model.bin'

    if args.ngrams != 2:
        model_path = model_path.parent / f'{model_path.stem}_{args.ngrams}.bin'

    if args.qualityfilter:
        model_path = model_path.parent / f"{model_path.stem}_qf.bin"

    if args.pipeline_step == 'model':
        if not model_path.exists() or args.overwrite_preprocess or args.retrain_model:
            fasttext_opts = {
                    'input': str(train_file),
                    'wordNgrams': args.ngrams,
                    'dim': 300,
                    'thread': args.num_proc,
                    'autotuneDuration': 1800,
                    'autotuneValidationFile': str(val_file)}

            fasttext_opts['pretrainedVectors'] = args.word_vectors

            print(fasttext_opts)

            model = fasttext.train_supervised(**fasttext_opts)
            model.save_model(str(model_path))

            n_samples, precision, recall = model.test(str(val_file))
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1:", 2 * (precision * recall) / (precision + recall))
            del model

    elif args.pipeline_step.startswith('predict'):
        predict_chunk(model_path, ds_dir, args.chunk_idx)
    elif args.pipeline_step == 'retrieve':
        retrieve_from_pile(model_path, args.num_to_retrieve, ds_dir)
    else:
        raise ValueError('not implemented')
