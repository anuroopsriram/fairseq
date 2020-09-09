#!/bin/bash

data=$1
job_dir=$2

echo $data
echo $job_dir

while true; do

lmweight=$(python3 -c  "import random; print(random.random() *  5 + 0.0001)")
wordscore=$(python3 -c "import random; print(random.random() *  5 + 0.0001)")
silweight=$(python3 -c "import random; print(random.random() * -5 - 0.0001)")
beamsize=200
beamscore=20
beamsizetoken=10

work_dir=$job_dir/"lmw=${lmweight}_ws=${wordscore}_silw=${silweight}"

if [ ! -d $work_dir ]; then
mkdir -p $work_dir

/private/home/abaevski/wav2letter/build/Decoder \
                  --emission_dir=$data                     \
                  --criterion=ctc                            \
                  --lexicon=/checkpoint/abaevski/emissions/w2l/wav2vec_q_10h/lexicon_lower.lst  \
                  --lm=/checkpoint/antares/icassp_2020_models/lms/word_gcnn_14B_antares.bin            \
                  --lm_vocab=/checkpoint/antares/icassp_2020_models/lms/word_gcnn_14B_antares.vocab \
                  --lmtype convlm                          \
                  --lm_memory=8000                         \
                  --beamsize=${beamsize}                             \
                  --beamthreshold=${beamscore}                           \
                  --beamsizetoken=${beamsizetoken}                       \
                  --nthread=60                                       \
                  --nthread_decoder=8                               \
                  --smearing=max                                    \
                  --lmweight=${lmweight}                            \
                  --silscore=${silweight}                          \
                  --wordscore=${wordscore} \
                  --rundir=$work_dir \
                  --logtostderr > $work_dir/result 2>&1
fi
done