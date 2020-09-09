#!/bin/bash

cp=$1
data=$2
task=$3
max_toks=$4
prefix=$5
lmscore=$6
wscore=$7
targets=$8
lm=$9
silweight=0

echo $cp
echo $data
echo $task
echo $max_toks
echo $prefix
echo $lmscore
echo $wscore
echo $silweight

cd ~abaevski/fairseq-py

jobdir=/checkpoint/$USER/speechbert_raw_sweeps/$prefix
mkdir -p $jobdir

sbatch --job-name speechbert_$prefix --gres gpu:1 --nodes 1 --ntasks-per-node 1 --cpus-per-task 8 --output $jobdir/stdout_%A_%a.log --error $jobdir/stderr_%A_%a.log --open-mode append --partition learnfair,dev --time 4320 --mem-per-cpu 40G --constraint volta32gb --wrap "
srun --job-name eval_${prefix}_dev_clean --output $jobdir/stdout_%A_%a.log --error $jobdir/stderr_%A_%a.log --open-mode append --unbuffered bash /private/home/abaevski/fairseq-py/sweeps/run_eval.sh $cp $data $task $max_toks $jobdir dev_clean $lmscore $wscore $silweight $targets $lm &
wait \$!"

sbatch --job-name speechbert_$prefix --gres gpu:1 --nodes 1 --ntasks-per-node 1 --cpus-per-task 8 --output $jobdir/stdout_%A_%a.log --error $jobdir/stderr_%A_%a.log --open-mode append --partition learnfair,dev --time 4320 --mem-per-cpu 40G --constraint volta32gb --wrap "
srun --job-name eval_${prefix}_dev_other --output $jobdir/stdout_%A_%a.log --error $jobdir/stderr_%A_%a.log --open-mode append --unbuffered bash /private/home/abaevski/fairseq-py/sweeps/run_eval.sh $cp $data $task $max_toks $jobdir dev_other $lmscore $wscore $silweight $targets $lm &
wait \$!"

sbatch --job-name speechbert_$prefix --gres gpu:1 --nodes 1 --ntasks-per-node 1 --cpus-per-task 8 --output $jobdir/stdout_%A_%a.log --error $jobdir/stderr_%A_%a.log --open-mode append --partition learnfair,dev --time 4320 --mem-per-cpu 40G --constraint volta32gb --wrap "
srun --job-name eval_${prefix}_test_clean --output $jobdir/stdout_%A_%a.log --error $jobdir/stderr_%A_%a.log --open-mode append --unbuffered bash /private/home/abaevski/fairseq-py/sweeps/run_eval.sh $cp $data $task $max_toks $jobdir test_clean $lmscore $wscore $silweight $targets $lm &
wait \$!"

sbatch --job-name speechbert_$prefix --gres gpu:1 --nodes 1 --ntasks-per-node 1 --cpus-per-task 8 --output $jobdir/stdout_%A_%a.log --error $jobdir/stderr_%A_%a.log --open-mode append --partition learnfair,dev --time 4320 --mem-per-cpu 40G --constraint volta32gb --wrap "
srun --job-name eval_${prefix}_test_other --output $jobdir/stdout_%A_%a.log --error $jobdir/stderr_%A_%a.log --open-mode append --unbuffered bash /private/home/abaevski/fairseq-py/sweeps/run_eval.sh $cp $data $task $max_toks $jobdir test_other $lmscore $wscore $silweight $targets $lm &
wait \$!"