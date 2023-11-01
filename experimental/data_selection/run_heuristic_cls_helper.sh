#!/bin/bash
set -x
source config.sh

mkdir -p $CACHE
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=100000000000
export TORCH_EXTENSIONS_DIR=$CACHE

task=$1
args=$2

mkdir -p ${OUTPUT_DIR}

python heuristic_cls_pipeline.py \
    --pile_path ${PILE_PATH}
    --ds_name ${task} \
    --output_dir ${OUTPUT_DIR} \
    --cache_dir ${CACHE} \
    --word_vectors ${WORD_VECTORS_PATH} \
    ${args}


