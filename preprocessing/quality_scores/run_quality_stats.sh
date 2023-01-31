#!/bin/bash

ds_path=$1
pile_path=$2

python compute_quality_stats.py --ds_path ${ds_path}

python merge_quality_scores.py --pile_path ${pile_path}
