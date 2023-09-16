#!/bin/bash

source config.sh

LOGDIR=logs/preprocessing/qualitystats
mkdir -p ${LOGDIR}

for SUBSET in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29; do
    jid=$(sbatch \
            --parsable \
            ${cluster_info} \
            --mem 24G \
            -c 16 \
            --output ${LOGDIR}/chunk_${SUBSET} \
            preprocessing/quality_scores/run_quality_stats.sh ${PILE_PATH}/chunked/${SUBSET}_128/${SUBSET}_128.json)
    echo -n "${jid} "
done


# validation data
jid=$(sbatch \
        --parsable \
        ${cluster_info} \
        --mem 24G \
        -c 16 \
        --output ${LOGDIR}/chunk_val \
        run_quality_stats.sh ${PILE_PATH}/chunked/VAL_128/val_128.json)
echo -n "${jid} "

