#!/bin/bash

function devsub {
      echo $*
      sbatch --job-name=$1 --output=$1-%j.out --error=$1-%j.out --nodes=1 --ntasks-per-node=1 --cpus-per-task=10 --gres=gpu:1 --signal=USR1@600 --open-mode=append --time=72:00:00 --partition=dev --wrap="srun $2"
}


#MODEL='logs/w2v.conformer.relpos.s2s.400k.ft/dim512.enclyrs17.lr0.0005.rpemb16.unlab/lr2e-05.lab.960h/checkpoint_best.pt'
#MODEL='logs/w2v.conformer.relpos.s2s.400k.ft/dim512.enclyrs17.lr0.0005.rpemb16.unlab/lr0.0001.lab.960h/checkpoint_best.pt'
MODEL='logs/w2v.conformer.relpos.s2s.400k.ft/dim512.enclyrs17.lr0.0005.rpemb16.unlab/lr0.0001.lab.960h/checkpoint_last.pt'

LM='/checkpoint/henryzhou7/wp_lm/transformer_raw3_adam_cosine2node/lr_1e-4_updatefreq_8/checkpoint_best.pt'
#LM='/checkpoint/abaevski/data/speech/libri/4-gram.bin'
#LM='/checkpoint/abaevski/models/libri_lms/translm2/checkpoint_best.pt'


BASECMD="python fairseq_cli/generate.py /checkpoint/henryzhou7/dataset/libri/960h/raw3/decoder \
  --task audio_pretraining --seed 1 --nbest 1 \
  --gen-subset dev_other --max-tokens 600000 \
  --path ${MODEL} --labels 10k --remove-bpe wordpiece \
  --quiet --beam 50 --scoring wer \
  --lm-path ${LM}"


for lmwt in 0 0.05 0.1 0.25 0.5 1 1.5; do
  for temp in 1; do
      CMD="${BASECMD} --lm-weight ${lmwt}  --temperature ${temp}"
      # echo $CMD
      devsub "s2s-decode-${lmwt}-${temp}" "${CMD}"
   done
done
