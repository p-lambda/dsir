import json
import numpy as np
from datasets import load_dataset
from argparse import ArgumentParser
import os
from pathlib import Path
from tqdm import tqdm

CHUNK_LENGTH=128

parser = ArgumentParser()
parser.add_argument('--input_dir', type=str)
parser.add_argument('--output_dir', type=str)
parser.add_argument('--input_filename', default="22.jsonl.zst", type=str)
parser.add_argument('--chunk_length', default=128, type=int)
parser.add_argument('--output_filename', default=None, type=str)
parser.add_argument('--cache_dir', type=str)


def chunk_examples(examples, chunk_length=CHUNK_LENGTH):
    chunks, metadata = [], []
    for sentence, meta in zip(examples['text'], examples['meta']):
        words = sentence.split(' ')
        curr_chunks = [' '.join(words[i:i + chunk_length]) for i in range(0, len(words), chunk_length)]
        chunks += curr_chunks
        metadata += [meta] * len(curr_chunks)
    return {'contents': chunks, 'metadata': metadata}

def add_id(examples, idx):
    examples['id'] = idx
    return examples

def main(args):
    CHUNK_LENGTH = args.chunk_length

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.output_filename is None:
        args.output_filename = args.input_filename.split('.')[0] + f'_{args.chunk_length}.json'

    print("Beginning dataset load")
    ds = load_dataset('json',
                      data_files=[f'{args.input_dir}/{args.input_filename}'],
                      cache_dir=args.cache_dir,
                      streaming=True)['train']

    column_names = list(next(iter(ds)).keys())

    ds = ds.map(chunk_examples, batched=True, remove_columns=column_names)

    print("Done loading dataset")
    ds = ds.map(add_id, batched=True, with_indices=True)

    print("Saving file")
    with open(Path(args.output_dir) / args.output_filename, 'w') as f:
        for ex in tqdm(iter(ds)):
            f.write(json.dumps(ex).strip() + '\n')

if __name__=="__main__":

    args = parser.parse_args()
    main(args)

