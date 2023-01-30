#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "${parent_path}"
# load global parameters: CACHE and ROOT_DIR
set -a
source ../config.sh
set +a

source ${VIRTUAL_ENV}/bin/activate
set -x
mkdir -p $CACHE
export HF_HOME=$CACHE
export TRANSFORMERS_CACHE=$CACHE
export HF_DATASETS_CACHE=$CACHE
export HF_DATASETS_IN_MEMORY_MAX_SIZE=0
export TORCH_EXTENSIONS_DIR=$CACHE

task=$1
args=$2

output_dir=${ROOT_DIR}/synthetic_pretraining/ngram_matching
mkdir -p ${output_dir}

export PYTHONHASHSEED=42

python is_pipeline.py \
    --ds_name ${task} \
    --output_dir ${output_dir} \
    --cache_dir ${CACHE} \
    ${args}
