#!/bin/zsh

cp=$1
data=$2
task=$3
max_toks=$4
job_dir=$5
split=$6
lmweight=$7
wordscore=$8
silweight=$9
targets=${10}
num_shards=${11}
normalize=${12}
lm=${13}

echo $cp
echo $data
echo $job_dir

beamsize=1500
beamscore=100

work_dir=$job_dir

if [ "$targets" != "ltr" ]; then
  bpe='@@ '
else
  bpe="letter"
fi

if [ "$normalize" = "true" ]; then
  normalize="--normalize"
else
  normalize=""
fi

if [ "$lm" != "" ]; then
  lm_model=$lm
  decoder=fairseqlm
  beamsize=500
else
  lm_model=/checkpoint/abaevski/data/speech/libri/4-gram.bin
  decoder=kenlm
fi

PYTHONPATH=/private/home/abaevski/fairseq-py-master python examples/speech_recognition/infer.py $data --task $task --seed 1 --nbest 1 --path $cp --gen-subset $split \
--results-path $work_dir --w2l-decoder $decoder --lm-model $lm_model \
--lexicon /checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw/lexicon_$targets.lst --beam $beamsize --beam-threshold $beamscore --beam-size-token 100 \
--lm-weight $lmweight --word-score $wordscore --sil-weight $silweight --criterion ctc  --labels $targets --max-tokens $max_toks --remove-bpe "$bpe" $normalize \
--shard-id $(expr ${SLURM_ARRAY_TASK_ID} - 1) --num-shards $num_shards
