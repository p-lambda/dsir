#!/bin/bash
set -x

source config.sh

TASK=$1
SUFFIX=$2
PORT=$3
PRETRAIN_DATA_PATH=$4
DO_PREPROCESS=${5:-"true"}
DO_PRETRAIN=${6:-"true"}
OTHER_ARGS=${7:-""}
LR=${8:-"5e-4"}


LOGDIR=logs/train
mkdir -p ${LOGDIR}

dependency=""
if [[ "${DO_PREPROCESS}" = "true" ]]; then
    jid=$(sbatch \
        --parsable \
        --mem=128g \
        ${cluster_info} \
        -c 8 \
        --output ${LOGDIR}/${TASK}_preprocess_${SUFFIX}.log \
        train/preprocess_general.sh ${TASK} ${PRETRAIN_DATA_PATH} ${CACHE} "${OTHER_ARGS}")
    echo -n "${jid} "
    dependency="--dependency afterok:${jid}"

fi

# pretrain
if [[ "${DO_PRETRAIN}" = "true" ]]; then
    # --mem=96g \
    jid=$(sbatch \
        --parsable \
        ${dependency} \
        ${cluster_info} \
        --gres=gpu:4 \
        -c 8 \
        --mem=64g \
        -t 14-0:00 \
        --output ${LOGDIR}/${TASK}_pretrain_${SUFFIX}.log \
        train/pretrain_general.sh ${TASK} ${PRETRAIN_DATA_PATH} "0,1,2,3" 4 ${TASK}_${SUFFIX} ${PORT} ${PRETRAIN_OUTPUT_DIR} ${CACHE} "${OTHER_ARGS}" ${LR})
    echo -n "${jid} "
    dependency="--dependency afterok:${jid}"
else
    dependency=""
fi
