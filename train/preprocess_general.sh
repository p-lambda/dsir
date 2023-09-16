#!/bin/bash
source config.sh

source ${VIRTUAL_ENV}/bin/activate

TASK=$1
PRETRAIN_DATA_PATH=$2
CACHE=$3
OTHER_ARGS=$4
MAXLEN=128

mkdir -p $CACHE
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export TORCH_EXTENSIONS_DIR=$CACHE
export TMPDIR=$CACHE




python run_pipeline.py \
    --pipeline_step preprocess \
    --preprocessing_num_workers 32 \
    --cache_dir ${CACHE} \
    --dataset_path ${PRETRAIN_DATA_PATH} \
    --max_ckpts_to_keep 3 \
    --max_length ${MAXLEN} \
    --pad_to_max_length \
    --model_name_or_path bert-base-uncased \
    --per_device_train_batch_size 256 \
    --per_device_eval_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --task_name $TASK \
    --save_final \
    --seed 42 \
    ${OTHER_ARGS}




