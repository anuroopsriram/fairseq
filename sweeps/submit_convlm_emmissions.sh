#!/bin/bash

data=$1
prefix=$2

echo $data
echo $prefix

cd ~abaevski/fairseq-py

jobdir=/checkpoint/$USER/speechbert_convlm_sweeps/$prefix
mkdir -p $jobdir

sbatch --job-name speechbert_$prefix --array=1-100 --gres gpu:8 --nodes 1 --ntasks-per-node 1 --cpus-per-task 64 --output $jobdir/stdout_%A_%a.log --error $jobdir/stderr_%A_%a.log --open-mode append --partition learnfair --comment icml --time 1500 --mem-per-cpu 6G --wrap "
srun --job-name spb_convlm_$prefix --output $jobdir/stdout_%A_%a.log --error $jobdir/stderr_%A_%a.log --open-mode append --unbuffered bash /private/home/abaevski/fairseq-py/sweeps/run_decode_w2l_emissions.sh $data $jobdir &
wait \$!"