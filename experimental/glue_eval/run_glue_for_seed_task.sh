#!/bin/bash
set -x

PRETRAINED_PATH=$1
TASK_NAME=$2
SEED=$3
EPOCHS=$4
LR=$5
BATCH_SIZE=$6
CACHE=$7
BASENAME=${8:-"bert-base-uncased"}
SAVE_PATH=${9:-${PRETRAINED_PATH}}
MAX_LEN=${10:-128}

mkdir -p $CACHE
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export TORCH_EXTENSIONS_DIR=$CACHE

export WANDB_PROJECT="glue"


FINETUNE_PATH=${SAVE_PATH}/finetune-runs
mkdir -p $FINETUNE_PATH

RUN=${TASK_NAME}_EPOCHS${EPOCHS}_BS${BATCH_SIZE}_LR${LR}_seed${SEED}
RUN_DIR=${FINETUNE_PATH}/${RUN}
mkdir -p $RUN_DIR


ACCUM=1

if [[ $(ls ${RUN_DIR}/eval_results*) ]]; then
  echo "${RUN} finished"
else
    python glue_eval/run_glue.py \
        --fp16 \
        --model_name_or_path $PRETRAINED_PATH \
        --tokenizer_name ${BASENAME} \
        --config_name ${BASENAME} \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --max_seq_length ${MAX_LEN} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --gradient_accumulation_steps ${ACCUM} \
        --num_train_epochs ${EPOCHS} \
        --run_name ${RUN_DIR} \
        --logging_steps 100 \
        --save_strategy epoch \
        --evaluation_strategy epoch \
        --learning_rate ${LR} \
        --seed ${SEED} \
        --output_dir ${RUN_DIR} \
        --logging_dir ${RUN_DIR}/tensorboard \
        --max_grad_norm 1.0 \
        --lr_scheduler_type polynomial \
        --weight_decay 0.1 \
        --load_best_model_at_end \
        --save_total_limit 1 \
        --adam_beta1 0.9 \
        --adam_beta2 0.98 \
        --adam_epsilon 1e-6 \
        --warmup_ratio 0.06 \
        --overwrite_output_dir
fi
