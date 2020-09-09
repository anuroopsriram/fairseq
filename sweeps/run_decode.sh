#!/bin/bash

cp=$1
data=$2
task=$3
max_toks=$4
job_dir=$5

echo $cp
echo $data
echo $job_dir

while true; do

lmweight=$(python3 -c  "import random; print(random.random() *  5 + 0.01)")
wordscore=$(python3 -c "import random; print(random.random() *  5 + 0.01)")
silweight=$(python3 -c "import random; print(random.random() * -5 - 0.01)")
beamsize=200
beamscore=40

work_dir=$job_dir/"lmw=${lmweight}_ws=${wordscore}_silw=${silweight}"

if [ ! -d $work_dir ]; then
mkdir -p $work_dir
echo $work_dir

if [ "$task" == "translation" ]; then

python examples/speech_recognition/infer.py $data --task $task --seed 1 --nbest 1 --path $cp --gen-subset dev_other \
--results-path $work_dir --w2l-decoder kenlm --kenlm-model /checkpoint/abaevski/data/speech/libri/4-gram.bin \
--lexicon /checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw/lexicon_ltr2.lst --beam $beamsize --beam-threshold $beamscore \
--lm-weight $lmweight --word-score $wordscore --sil-weight $silweight --criterion ctc  --target-lang ltr \
--max-tokens $max_toks --left-pad-source "False" --left-pad-target "False" --max-source-positions 2048 --source-lang src
/private/home/abaevski/sctk-2.4.10/bin/sclite -r $work_dir/ref.word-checkpoint_best.pt-dev_other.txt \
-h $work_dir/hypo.word-checkpoint_best.pt-dev_other.txt  -i rm -o all stdout > $work_dir/report

else

python examples/speech_recognition/infer.py $data --task $task --seed 1 --nbest 1 --path $cp --gen-subset dev_other \
--results-path $work_dir --w2l-decoder kenlm --kenlm-model /checkpoint/abaevski/data/speech/libri/4-gram.bin \
--lexicon /checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw/lexicon_ltr2.lst --beam $beamsize --beam-threshold $beamscore \
--lm-weight $lmweight --word-score $wordscore --sil-weight $silweight --criterion ctc  --labels ltr --max-tokens $max_toks
/private/home/abaevski/sctk-2.4.10/bin/sclite -r $work_dir/ref.word-checkpoint_best.pt-dev_other.txt \
-h $work_dir/hypo.word-checkpoint_best.pt-dev_other.txt  -i rm -o all stdout > $work_dir/report

fi

if [[ $? != 0 ]]; then
    rm -rf $work_dir
    break
fi

fi
done