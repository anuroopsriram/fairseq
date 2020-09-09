#!/bin/bash

cp=$1
data=$2
output_dir=$3
split=$4
num_shards=$5
labels=$6

python ./featurize-vq.py --checkpoint $cp --output-dir $output_dir --data-dir $data --splits $split --extension tsv --shard $(expr ${SLURM_ARRAY_TASK_ID} - 1) --num-shards $num_shards --labels $labels
