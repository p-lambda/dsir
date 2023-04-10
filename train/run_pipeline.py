# Adapted from https://github.com/yaoxingcheng/TLM and HuggingFace example

import argparse
import logging
import math
import os, sys
import random
import torch
import datasets
from datasets import load_dataset, load_metric, load_from_disk, DatasetDict, concatenate_datasets
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from pathlib import Path
import wandb

import transformers
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    DataCollatorWithPadding,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version
from collator import DataCollatorForLanguageModeling
from model import BertForMaskedLM
from trainer import PretrainTrainer

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer model")
    # about active sampling
    parser.add_argument(
        "--tokenizer_file",
        type=str,
        default=None,
        help="The name of the tokenizer file or path"
    )
    parser.add_argument(
        "--pipeline_step",
        type=str,
        default='preprocess',
        help="step of the pipeline - preprocess, pretrain"
    )
    parser.add_argument(
        "--save_final",
        action="store_true",
        help="save the final checkpoint",
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help="training from scratch",
    )
    parser.add_argument(
        "--from_ckpt",
        type=str,
        default=None,
        help="restore the model training process from a checkpoint",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="path to the pretraining dataset"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default='/scr',
        help="path to cache directory"
    )
    parser.add_argument(
        "--max_ckpts_to_keep",
        type=int,
        default=3,
        help="Number of checkpoints to keep"
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="Number of preprocessors"
    )
    parser.add_argument(
        "--steps_to_log",
        type=int,
        default=None,
        help="Num steps to log training info"
    )
    parser.add_argument(
        "--steps_to_eval",
        type=int,
        default=None,
        help="Num steps to evaluate on the dev set"
    )
    parser.add_argument(
        "--steps_to_save",
        type=int,
        default=None,
        help="Num steps to save the checkpoint"
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Probability of masking"
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="Adam beta 1"
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="Adam beta 2"
    )
    parser.add_argument(
        "--adam_eps",
        type=float,
        default=1e-8,
        help="Adam epsilon"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.0,
        help="Max gradient norm"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the task to train on.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default=None,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=None, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=0,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        # choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=float, default=10000, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--cuda_devices", type=str, default='0', help="visible cuda devices."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    args = parser.parse_args()
    # Sanity checks
    if args.task_name is None:
        raise ValueError("Need a task name.")

    if args.model_name_or_path is None:
        assert args.from_scratch, "no model name or path is provided but trying to initialize from a pre-trained weight"

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args"), "w") as f:
            for arg in vars(args):
                f.write(f"{arg}: {getattr(args, arg)}\n")

    return args

def get_logger(args, accelerator=None):
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator is not None:
        logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    if args.output_dir is not None:
        logfile = os.path.join(args.output_dir, "log")
        if accelerator is not None and accelerator.is_main_process:
            if os.path.exists(logfile):
                os.remove(logfile)
            os.mknod(logfile)
            fh = logging.FileHandler(logfile, mode='w')
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    if accelerator is None:
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    else:
        logger.setLevel(logging.INFO if accelerator.is_main_process else logging.ERROR)
        if accelerator.is_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    return logger


def get_dataset(args, preprocessed_cache):
    data_files = {}
    data_files["train"] = args.dataset_path
    extension = Path(args.dataset_path).name.split(".")[-1]
    if extension == "txt":
        extension = "text"
    elif extension == "jsonl":
        extension = "json"
    else:
        extension = "json"

    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=f"{args.cache_dir}/cache")
    return raw_datasets


def preprocess(args, raw_datasets, tokenizer, logger, preprocessed_cache):
    logger.info("preprocessing datasets")
    if raw_datasets is not None:
        column_names = raw_datasets["train"].column_names
        text_column_name = "contents" if "contents" in column_names else column_names[0]
    else:
        text_column_name = "contents"

    padding = "max_length"
    def tokenize_function(examples):
        # Remove empty lines
        examples[text_column_name] = [
            line for line in examples[text_column_name] if line is not None and len(line) > 0 and not line.isspace()
        ]

        tokenized_examples = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=args.max_length,
            return_special_tokens_mask=True,
        )
        remove_cols = set(column_names)
        for k in examples:
            if k not in tokenized_examples and k not in {text_column_name, "rank", "ids", "id"} and k not in remove_cols:
                tokenized_examples[k] = examples[k][:len(examples[text_column_name])]

        return tokenized_examples

    num_cpus = len(os.sched_getaffinity(0))
    num_shards = 8
    tokenized_datasets_ls = []
    for shard_i in range(num_shards):
        preprocessed_cache.mkdir(exist_ok=True)
        preprocessed_cache_i = preprocessed_cache / f"shard_{shard_i}"
        if not preprocessed_cache_i.exists():
            raw_datasets_shard = raw_datasets["train"].shard(
                    num_shards=num_shards, index=shard_i).flatten_indices()
            logger.info(f"Processing shard {shard_i}")
            tokenized_datasets_i = raw_datasets_shard.map(
                    tokenize_function,
                    batched=True,
                    batch_size=100,
                    num_proc=num_cpus//2,
                    remove_columns=raw_datasets_shard.column_names,
                    desc="Running tokenizer on dataset line_by_line",
                )
            logger.info(f"Saving shard {shard_i} to disk")
            tokenized_datasets_i.save_to_disk(str(preprocessed_cache_i))

    for shard_i in range(num_shards):
        assert preprocessed_cache_i.exists()
        tokenized_datasets_i = load_from_disk(str(preprocessed_cache_i))
        tokenized_datasets_ls.append(tokenized_datasets_i)
    tokenized_datasets = concatenate_datasets(tokenized_datasets_ls)

    return tokenized_datasets, text_column_name

def get_model(args, load_model=True):
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.pipeline_step == 'preprocess':
        load_model = False

    if args.model_name_or_path and not args.from_scratch:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(args.config_dir)

        if load_model:
            config = AutoConfig.from_pretrained(args.model_name_or_path)

            # set_cls = True if not args.from_scratch else False
            if args.model_name_or_path == 'bert-base-uncased' or args.model_name_or_path == 'bert-large-uncased':
                model = BertForMaskedLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                )
            else:
                model = RobertaForMaskedLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                )
    elif args.model_name_or_path and args.from_scratch:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        if load_model:
            if args.model_name_or_path == 'bert-base-uncased' or args.model_name_or_path == 'bert-large-uncased':
                model = BertForMaskedLM(config)
            else:
                model = RobertaForMaskedLM(config)
    else:
        config = AutoConfig.from_pretrained(args.config_dir)
        tokenizer = AutoTokenizer.from_pretrained(args.config_dir)
        if load_model:
            if args.model_name_or_path == 'bert-base-uncased' or args.model_name_or_path == 'bert-large-uncased':
                model = BertForMaskedLM(config)
            else:
                model = RobertaForMaskedLM(config)

    if load_model:
        model.set_args(args)
    else:
        model = None
    return tokenizer, model


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    os.environ['NVIDIA_VISIBLE_DEVICES'] = args.cuda_devices

    preprocessed_cache = Path(args.dataset_path).parent / 'preprocessed_cache' / f"{args.task_name}_{Path(args.dataset_path).parent.name}_{Path(args.dataset_path).stem}"

    preprocessed_cache.parent.mkdir(exist_ok=True, parents=True)

    if args.pipeline_step == 'preprocess':
        logger = get_logger(args)
        tokenizer, model = get_model(args, load_model=False)

        raw_dataset = get_dataset(args, preprocessed_cache)
        dataset, text_column_name = preprocess(args, raw_dataset, tokenizer, logger, preprocessed_cache)

    elif args.pipeline_step == 'pretrain':
        accelerator = Accelerator(fp16=True)
        args.device = accelerator.device
        logger = get_logger(args, accelerator)
        tokenizer, model = get_model(args)
        if accelerator.is_main_process:
            wandb.init(project="pretrain", name=f"{args.pipeline_step}_{args.task_name}_{Path(args.output_dir).name}_{args.seed}")


        raw_dataset = None
        dataset, text_column_name = preprocess(args, raw_dataset, tokenizer, logger, preprocessed_cache=preprocessed_cache)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=args.mlm_probability)
        dataloader = DataLoader(
            dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size, num_workers=0
        )

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps)

        # Prepare everything with `accelerator`.
        model, optimizer, dataloader = accelerator.prepare(
            model, optimizer, dataloader
        )
        num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None or args.max_train_steps == 0:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=int(args.num_warmup_steps),
            num_training_steps=args.max_train_steps,
        )

        trainer = PretrainTrainer(
            args=args,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            dataloader=dataloader,
            logger=logger,
            accelerator=accelerator,
            from_checkpoint=args.from_ckpt,
            tokenizer=tokenizer,
            max_grad_norm=args.max_grad_norm,
        )
        trainer.train()

if __name__ == "__main__":
    main()
