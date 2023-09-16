#!/bin/bash
set -x

source config.sh

source ${VIRTUAL_ENV}/bin/activate

task=$1
run_name=$2
other_args=$3
num_to_retrieve=${4:-25000000}

LOGDIR=logs/preprocessing/heuristic_cls/${run_name}
mkdir -p ${LOGDIR}

NUM_CHUNKS=29

jid=$(sbatch \
        --parsable \
        --mem 32G \
        ${cluster_info} \
        -c 4 \
        --output ${LOGDIR}/train_fasttext_model \
        preprocessing/run_heuristic_cls_helper.sh ${task} "--pipeline_step model ${other_args} --num_proc 4")
echo -n "${jid} "

dependency="--dependency afterok"
for CHUNK_IDX in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29;
do
    predict_jid=$(sbatch \
            --parsable \
            ${cluster_info} \
            --dependency ${jid} \
            --mem 20G \
            -c 1 \
            --output ${LOGDIR}/predict_${CHUNK_IDX} \
            preprocessing/run_heuristic_cls_helper.sh ${task} "--pipeline_step predict --chunk_idx ${CHUNK_IDX} --num_chunks ${NUM_CHUNKS} ${other_args} --num_proc 1")
    echo -n "${predict_jid} "
    dependency="${dependency}:${predict_jid}"
done

jid=$(sbatch \
        --parsable \
        --mem 100G \
        ${cluster_info} \
        ${dependency} \
        -c 4 \
        --output ${LOGDIR}/retrieve_${num_to_retrieve} \
        preprocessing/run_heuristic_cls_helper.sh ${task} "--pipeline_step retrieve ${other_args} --num_chunks ${NUM_CHUNKS} --num_proc 4 --num_to_retrieve ${num_to_retrieve}")
echo -n "${jid} "

