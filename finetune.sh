#!/bin/bash
# example:
# bash examples/robust_wav2vec/finetune.sh base_ls-10h test +ckpt=ls_800k_64g +lm=ls_4gram
set -eu

config_name=$1
launcher=$2

if [ $launcher == "test" ]; then
  launcher_args="+run=local"
elif [ $launcher == "loc" ]; then
  launcher_args="hydra/launcher=submitit_local +run=submitit_loc"
elif [ $launcher == "dev" ]; then
  launcher_args="hydra/launcher=submitit_slurm +run=submitit_dev"
elif [ $launcher == "reg" ]; then
  launcher_args="hydra/launcher=submitit_slurm +run=submitit_reg"
elif [ $launcher == "reg24" ]; then
  launcher_args="hydra/launcher=submitit_slurm +run=submitit_reg_24gpu"
else
  echo "invalid mode ($launcher)" && exit 1;
fi

shift
shift

set -x
PYTHONPATH=. python fairseq_cli/hydra_train.py \
  -m --config-dir ~/wav2vec2_robust/fairseq-py-dev/users/wnhsu/robust_w2v/dev_202101/hydra/conf/finetune \
  --config-name $config_name $launcher_args $@

