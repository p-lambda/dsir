#!/bin/bash

source config.sh

LOGDIR=logs/preprocess
mkdir -p ${LOGDIR}

for SUBSET in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29; do
     jid=$(sbatch \
             --parsable \
             ${cluster_info} \
             --mem 48G \
             -c 16 \
             --output ${LOGDIR}/chunk_${SUBSET} \
             preprocessing/run.sh "--input_filename ${SUBSET}.jsonl.zst --chunk_length 128 --input_dir ${PILE_PATH} --output_dir ${PILE_PATH}/chunked/${SUBSET}_128 --cache_dir ${CACHE}}")
     echo -n "${jid} "
done

# validation data
 jid=$(sbatch \
         --parsable \
         ${cluster_info} \
         --mem 48G \
         -c 16 \
         --output ${LOGDIR}/chunk_val \
         preprocessing/run.sh "--input_filename val.jsonl.zst --chunk_length 128 --input_dir ${PILE_PATH} --output_dir ${PILE_PATH}/chunked/VAL_128 --cache_dir ${CACHE}}")
 echo -n "${jid} "

