#!/bin/bash

CACHE='/path/to/cachedir'
ROOT_DIR='/path/to/dsir/experimental'
VIRTUAL_ENV='/path/to/.env'
PILE_PATH='/path/to/pile'
DSIR_OUTPUT_DIR='/path/to/outputdir'
PRETRAIN_OUTPUT_DIR='/path/to/model_outputdir'
WORD_VECTORS_PATH='/path/to/pretrained_fasttext_wordvecs.vec'
# Slurm
cluster_info='--partition <PARTITION_NAME>'

source ${VIRTUAL_ENV}/bin/activate
