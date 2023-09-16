#!/bin/bash
source config.sh


task=$1
run_name=$2
other_args=$3
num_to_retrieve=${4:-25000000}

LOGDIR=logs/data_selection/dsir/${run_name}
mkdir -p ${LOGDIR}

NUM_CHUNKS=29
CHUNK_IDX=01
predict_jid=$(sbatch \
        --parsable \
        --mem 5G \
        ${cluster_info} \
        -c 2 \
        --output ${LOGDIR}/prepare_${CHUNK_IDX} \
        data_selection/run_dsir_helper.sh ${task} "--pipeline_step prepare --chunk_idx ${CHUNK_IDX} --num_chunks ${NUM_CHUNKS} ${other_args} --num_proc 16")

dependency="--dependency afterok"
for CHUNK_IDX in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29;
do
    predict_jid=$(sbatch \
            --parsable \
            --mem 5G \
            ${cluster_info} \
            -c 2 \
            --output ${LOGDIR}/predict_${CHUNK_IDX} \
            data_selection/run_dsir_helper.sh ${task} "--pipeline_step importance_weights --chunk_idx ${CHUNK_IDX} --num_chunks ${NUM_CHUNKS} ${other_args} --num_proc 16")
    echo -n "${predict_jid} "
    dependency="${dependency}:${predict_jid}"
done

jid=$(sbatch \
        --parsable \
        ${cluster_info} \
        --mem 48G \
        ${dependency} \
        -c 4 \
        --output ${LOGDIR}/retrieve_${num_to_retrieve} \
        run_dsir_helper.sh ${task} "--pipeline_step resample ${other_args} --num_proc 8 --num_to_retrieve ${num_to_retrieve}")
echo -n "${jid} "

