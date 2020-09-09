#!/bin/bash

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
lm=${11}

echo $cp
echo $data
echo $job_dir

beamsize=1500
beamscore=100

work_dir=$job_dir/$split

if [ ! -d $work_dir ]; then
mkdir -p $work_dir
echo $work_dir

if [ "$task" == "translation" ]; then

python examples/speech_recognition/infer.py $data --task $task --seed 1 --nbest 1 --path $cp --gen-subset $split \
--results-path $work_dir --w2l-decoder kenlm --kenlm-model /checkpoint/abaevski/data/speech/libri/4-gram.bin \
--lexicon /checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw/lexicon_ltr2.lst --beam $beamsize --beam-threshold $beamscore \
--lm-weight $lmweight --word-score $wordscore --sil-weight $silweight --criterion ctc  --target-lang ltr \
--max-tokens $max_toks --left-pad-source "False" --left-pad-target "False" --max-source-positions 2048 --source-lang src
/private/home/abaevski/sctk-2.4.10/bin/sclite -r $work_dir/ref.word-checkpoint_best.pt-$split.txt \
-h $work_dir/hypo.word-checkpoint_best.pt-$split.txt  -i rm -o all stdout > $work_dir/report

else

if [ "$targets" != "ltr" ]; then
  bpe='@@ '
else
  bpe=""
fi

if [ "$lm" != "" ]; then
  lm_model=$lm
  decoder=fairseqlm
  beamsize=500
else
  lm_model=/checkpoint/abaevski/data/speech/libri/4-gram.bin
  decoder=kenlm
fi

python examples/speech_recognition/infer.py $data --task $task --seed 1 --nbest 1 --path $cp --gen-subset $split \
--results-path $work_dir --w2l-decoder $decoder --lm-model $lm_model \
--lexicon /checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw/lexicon_$targets.lst --beam $beamsize --beam-threshold $beamscore --beam-size-token 100 \
--lm-weight $lmweight --word-score $wordscore --sil-weight $silweight --criterion ctc  --labels $targets --max-tokens $max_toks --remove-bpe "$bpe"
/private/home/abaevski/sctk-2.4.10/bin/sclite -r $work_dir/ref.word-checkpoint_best.pt-$split.txt \
-h $work_dir/hypo.word-checkpoint_best.pt-$split.txt  -i rm -o all stdout > $work_dir/report

fi

fi
