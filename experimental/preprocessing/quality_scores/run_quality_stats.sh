#!/bin/bash

DS_PATH=$1

python preprocessing/quality_scores/compute_quality_stats.py --ds_path ${DS_PATH}
