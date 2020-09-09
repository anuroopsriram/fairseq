#!/bin/bash

cp=$1
data=$2
output_dir=$3
split=$4
num_shards=$5
labels=$6

echo $cp
echo $data
echo $output_dir
echo $split
echo $num_shards

cd ~abaevski/fairseq-py

mkdir -p $output_dir

sbatch --job-name sharded_quantize -x learnfair0610 --array=1-$num_shards --gres gpu:1 --nodes 1 --ntasks-per-node 1 --cpus-per-task 8 --output $output_dir/stdout_%A_%a.log --error $output_dir/stderr_%A_%a.log --open-mode append --partition dev,learnfair --comment icml --time 2500 --mem-per-cpu 15G --wrap "
srun --job-name sharded_quantize --output $output_dir/stdout_%A_%a.log --error $output_dir/stderr_%A_%a.log --open-mode append --unbuffered bash sweeps/run_sharded_quantize.sh $cp $data $output_dir $split $num_shards $labels &
wait \$!"