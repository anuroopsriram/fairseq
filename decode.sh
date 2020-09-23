#!/bin/bash

NAME=""

DATA=""
LEXICON="/checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw/lexicon_ltr.lst"
LOGDIR=""
MAXTOKS=4000000

DECODER="kenlm"
LM="/checkpoint/abaevski/data/speech/libri/4-gram.bin"
BEAM=500
PREFIX="${NAME}.kelm"
CONSTRAINT=""

#DECODER="fairseqlm"
#LM="/checkpoint/abaevski/models/libri_lms/translm2/checkpoint_best.pt"
#BEAM=50
#PREFIX="${NAME}.translm"
#CONSTRAINT="volta32gb"


python decode_automl.py \
  -d $DATA \
  -s $LOGDIR \
  -l $LEXICON \
  -e $LOGDIR/emissions.txt \
  -m $MAXTOKS \
  --decoder $DECODER \
  --lm $LM \
  --beam $BEAM \
  --remove-bpe "letter" \
  --prefix $PREFIX \
  -g 8 \
  --partition dev \
  --constraint $CONSTRAINT \
  --num-parallel-jobs 2 \
  --num-runs 128

