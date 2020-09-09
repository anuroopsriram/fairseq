#!/bin/bash

cp=$1
data=$2
task=$3
max_toks=$4
prefix=$5

echo $cp
echo $data
echo $task
echo $max_toks
echo $prefix

cd ~abaevski/fairseq-py

jobdir=/checkpoint/$USER/speechbert_sweeps_1h/$prefix
mkdir -p $jobdir

sbatch --job-name speechbert_$prefix --array=1-100 --gres gpu:1 --nodes 1 --ntasks-per-node 1 --cpus-per-task 8 --output $jobdir/stdout_%A_%a.log --error $jobdir/stderr_%A_%a.log --open-mode append --partition priority,learnfair --comment icassp --time 1000 --mem-per-cpu 6G --wrap "
srun --job-name speechbert_4gram_$prefix --output $jobdir/stdout_%A_%a.log --error $jobdir/stderr_%A_%a.log --open-mode append --unbuffered bash /private/home/abaevski/fairseq-py/sweeps/run_decode.sh $cp $data $task $max_toks $jobdir &
wait \$!"