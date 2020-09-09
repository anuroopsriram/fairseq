#!/bin/bash

cp=$1
data=$2
task=$3
max_toks=$4
prefix=$5
emissions=$6

echo $cp
echo $data
echo $task
echo $max_toks
echo $prefix
echo $emissions

cd ~abaevski/fairseq-py

jobdir=/checkpoint/$USER/speechbert_sweeps_1h_em2/$prefix
mkdir -p $jobdir

sbatch --job-name speechbert_$prefix --array=1-150 --gres gpu:0 --nodes 1 --ntasks-per-node 1 --cpus-per-task 8 --output $jobdir/stdout_%A_%a.log --error $jobdir/stderr_%A_%a.log --open-mode append --partition dev,learnfair,priority --comment acl_arxiv --time 1000 --mem-per-cpu 6G --wrap "
srun --job-name speechbert_4gram_$prefix --output $jobdir/stdout_%A_%a.log --error $jobdir/stderr_%A_%a.log --open-mode append --unbuffered bash /private/home/abaevski/fairseq-py/sweeps/run_decode_emissions.sh $cp $data $task $max_toks $jobdir $emissions &
wait \$!"