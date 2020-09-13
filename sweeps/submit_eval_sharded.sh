#!/bin/zsh

cp=$1
data=$2
task=$3
max_toks=$4
prefix=$5
lmscore=$6
wscore=$7
targets=$8
num_shards=$9
normalize=${10}
lm=${11}
splits=${12}

silweight=0

echo $cp
echo $data
echo $task
echo $max_toks
echo $prefix
echo $lmscore
echo $wscore
echo $silweight

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
jobdir=/checkpoint/$USER/speechbert_raw_sweeps_new/$prefix
mkdir -p $jobdir

if [ -z "$splits" ]; then
  splits=(dev_clean dev_other test_clean test_other)
else
  IFS=',' read -r -a splits <<< "$splits"
fi

for split in "${splits[@]}"; do
  echo $split
#  jobid=$(sbatch --job-name eval_${prefix}_$split --parsable --array=1-$num_shards --gres gpu:1 --nodes 1 --ntasks-per-node 1 --cpus-per-task 4 --output $jobdir/stdout_%A_%a.log --error $jobdir/stderr_%A_%a.log --open-mode append --partition learnfair,dev --time 4320 --mem 30G --wrap "
#  srun --job-name eval_${prefix}_$split --output $jobdir/stdout_%A_%a.log --error $jobdir/stderr_%A_%a.log --open-mode append --unbuffered zsh $script_dir/run_eval_sharded.sh $cp $data $task $max_toks $jobdir $split $lmscore $wscore $silweight $targets $num_shards $normalize $lm &
#  wait \$!")

  echo "$script_dir/run_eval_sharded.sh $cp $data $task $max_toks $jobdir $split $lmscore $wscore $silweight $targets $num_shards $normalize $lm"

#  echo $jobid

#  sbatch --job-name score_${prefix}_$split --dependency=afterok:$jobid --nodes 1 --ntasks-per-node 1 --cpus-per-task 1 --output $jobdir/stdout_%A_%a.log --error $jobdir/stderr_%A_%a.log --open-mode append --partition learnfair,dev --time 20 --mem-per-cpu 1G --wrap "
#    srun --job-name eval_${prefix}_$split --output $jobdir/stdout_%A_%a.log --error $jobdir/stderr_%A_%a.log --open-mode append --unbuffered zsh /private/home/abaevski/fairseq-py/sweeps/score_evals.sh $jobdir $split $num_shards $cp &
#    wait \$!"

  echo "sweeps/score_evals.sh $jobdir $split $num_shards $cp"
done
