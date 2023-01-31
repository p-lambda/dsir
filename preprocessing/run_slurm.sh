#!/bin/bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "${parent_path}"
# load global parameters
source ../config.sh

mkdir -p logs

for SUBSET in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29; do
     jid=$(sbatch \
             --parsable \
             --partition john \
             --exclude john[12-17],john5,john3 \
             --mem 48G \
             -c 16 \
             --output logs/chunk_${SUBSET} \
             run.sh "--input_filename ${SUBSET}.jsonl.zst --chunk_length 128 --input_dir ${PILE_PATH} --output_dir ${PILE_PATH}/chunked/${SUBSET}_128 --cache_dir ${CACHE}}" ${VIRTUAL_ENV} ${CACHE} ${PILE_PATH})
     echo -n "${jid} "
done


