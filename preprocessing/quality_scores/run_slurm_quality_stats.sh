#!/bin/bash

mkdir -p logs/qualitystats

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "${parent_path}"
# load global parameters
source ../../config.sh

for SUBSET in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29; do
    jid=$(sbatch \
            --parsable \
            ${cluster_info} \
            --mem 24G \
            -c 16 \
            --output logs/qualitystats/chunk_${SUBSET} \
            run_quality_stats.sh ${PILE_PATH}/chunked/${SUBSET}_128/${SUBSET}_128.json ${pile_path})
    echo -n "${jid} "
done



