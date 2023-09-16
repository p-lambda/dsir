#!/bin/bash
source config.sh

source ${VIRTUAL_ENV}/bin/activate


TASK=$1
PRETRAIN_DATA_PATH=$2
CUDA_DEVICES=$3
NUM_GPUS=$4
SAVENAME=$5
PORT=$6
PRETRAIN_OUTPUT_DIR=$7
CACHE=$8
OTHER_ARGS=$9
LR=${10:-"5e-4"}

mkdir -p $CACHE
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export TORCH_EXTENSIONS_DIR=$CACHE
export TMPDIR=$CACHE


# We expect 4 GPUs / RTX 3090
BATCH_SIZE=64
GRAD_ACCUM=16

mkdir -p $PRETRAIN_OUTPUT_DIR

WD=0.01
WARMUP=3000

OUTPUT_PATH=${PRETRAIN_OUTPUT_DIR}/${SAVENAME}_pretrain
MAXLEN=128


accelerate launch \
    --config_file ./accelerate_config.yaml \
   --main_process_port ${PORT} \
   --num_processes ${NUM_GPUS} \
    run_pipeline.py \
    --pipeline_step pretrain \
    --cache_dir ${CACHE} \
    --dataset_path ${PRETRAIN_DATA_PATH} \
    --max_train_steps 50000 \
    --steps_to_eval 1000 \
    --steps_to_save 12500 \
    --steps_to_log 100 \
    --max_ckpts_to_keep 4 \
    --max_length ${MAXLEN} \
    --pad_to_max_length \
    --model_name_or_path bert-base-uncased \
    --output_dir $OUTPUT_PATH \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size 256 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --cuda_devices $CUDA_DEVICES \
    --task_name $TASK \
    --save_final \
    --weight_decay $WD \
    --learning_rate $LR \
    --num_warmup_steps $WARMUP \
    --seed 42 \
    ${OTHER_ARGS}


