#!/bin/bash

data=$1
prefix=$2

echo $data
echo $prefix

cd ~abaevski/fairseq-py

jobdir=/checkpoint/$USER/speechbert_sweeps_w2l/$prefix
mkdir -p $jobdir

sbatch --job-name speechbert_$prefix --array=1-300 --gres gpu:0 --nodes 1 --ntasks-per-node 1 --cpus-per-task 8 --output $jobdir/stdout_%A_%a.log --error $jobdir/stderr_%A_%a.log --open-mode append --partition priority,learnfair --comment icassp --time 1000 --mem-per-cpu 6G --wrap "
srun --job-name speechbert_4gram_$prefix --output $jobdir/stdout_%A_%a.log --error $jobdir/stderr_%A_%a.log --open-mode append --unbuffered bash /private/home/abaevski/fairseq-py/sweeps/run_decode_w2l.sh $data $jobdir &
wait \$!"