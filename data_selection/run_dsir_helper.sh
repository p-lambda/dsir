#!/bin/bash

set -x

source config.sh

source ${VIRTUAL_ENV}/bin/activate
mkdir -p $CACHE
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export TORCH_EXTENSIONS_DIR=$CACHE

task=$1
args=$2

output_dir=${DSIR_OUTPUT_DIR}
mkdir -p ${output_dir}

# important! set this environment variable to make sure the hash
# function is consistent across different machines.
export PYTHONHASHSEED=42

python dsir_pipeline.py \
    --pile_path ${PILE_PATH} \
    --ds_name ${task} \
    --output_dir ${output_dir} \
    --cache_dir ${CACHE} \
    ${args}
