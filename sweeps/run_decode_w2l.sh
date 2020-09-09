#!/bin/bash

data=$1
job_dir=$2

echo $data
echo $job_dir

while true; do

lmweight=$(python3 -c  "import random; print(random.random() *  5 + 0.01)")
wordscore=$(python3 -c "import random; print(random.random() *  5 + 0.01)")
silweight=$(python3 -c "import random; print(random.random() * -5 - 0.01)")
beamsize=200
beamscore=15

work_dir=$job_dir/"lmw=${lmweight}_ws=${wordscore}_silw=${silweight}"

if [ ! -d $work_dir ]; then
mkdir -p $work_dir

/private/home/abaevski/wav2letter-experimental/build/Decoder --test dev_other \
                  --emission_dir=$data/ctc                     \
                  --targetdir=$data/dev_other        \
                  --lexicon=/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw/lexicon_ltr_nopipe.lst       \
                  --lm=/checkpoint/abaevski/data/speech/libri/4-gram.bin            \
                  --beamsize=200                             \
                  --beamscore=15                           \
                  --nthread=8                                       \
                  --nthread_decoder=8                               \
                  --smearing=max                                    \
                  --lmweight=${lmweight}                            \
                  --silweight=${silweight}                          \
                  --wordscore=${wordscore} \
                  --rundir=$work_dir \
                  --datadir=$data \
                  --tokensdir=$data \
                  --logtostderr > $work_dir/result 2>&1

fi
done