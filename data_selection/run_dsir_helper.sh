#!/bin/bash

set -x

source ${VIRTUAL_ENV}/bin/activate
mkdir -p $CACHE
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export TORCH_EXTENSIONS_DIR=$CACHE

pile_path=$1
task=$2
args=$3

output_dir=${DSIR_OUTPUT_DIR}
mkdir -p ${output_dir}

export PYTHONHASHSEED=42

python dsir_pipeline.py \
    --pile_path ${pile_path} \
    --ds_name ${task} \
    --output_dir ${output_dir} \
    --cache_dir ${CACHE} \
    ${args}
