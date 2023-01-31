#!/bin/bash
num_buckets=10000
ngrams=2

#### target = books and wiki.
#### for the wiki_and_books target, we add 4M additional examples from the wiki and books related subsets of the Pile.
#### pack_every_2_examples reduces the number of examples by a factor of 2

TARGET='wiki_and_books'
run_name=${TARGET}_all_ngrammatching_ngrams${ngrams}_buckets${num_buckets}_qf
bash run_dsir.sh ${TARGET} ${run_name} " --qualityfilter --ngrams ${ngrams} --num_buckets ${num_buckets} --pack_every_2_examples" 98400000

