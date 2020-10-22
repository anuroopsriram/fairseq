#!/bin/bash

NAME="dev_other"

DATA="/checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw/"
LEXICON="/checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw/lexicon_ltr2.lst"
#LOGDIR="logs/w2v.conformer.relpos.400k.ft/dim512.enclyrs17.lr0.0005.rpemb16.unlab/lr6e-05.lab.960h/infer/"
#LOGDIR="logs/w2v.conformer.relpos.600k.16nd.ft/lr0.001.unlab/lr6e-05.lab.960h/infer"
LOGDIR="$1/infer/"
MAXTOKS=4000000

DECODER="kenlm"
LM="/checkpoint/abaevski/data/speech/libri/4-gram.bin"
BEAM=500
PREFIX="${NAME}"
CONSTRAINT="volta32gb"

echo python decode_automl.py \
  -d $DATA \
  --gen-subset dev_other \
  --log-dir ${LOGDIR}/${DECODER} \
  -l $LEXICON \
  -e $LOGDIR/emissions_dev_other.npy \
  -m $MAXTOKS \
  --decoder $DECODER \
  --lm $LM \
  --beam $BEAM \
  --remove-bpe letter \
  --prefix $PREFIX \
  -g 1 -j 8 --num-runs 128 \
  --partition dev,learnfair \
  --constraint $CONSTRAINT \
  --lmwt_min 1 --lmwt_max 4 --wrdsc_min -4 --wrdsc_max 1


#DECODER="fairseqlm"
#LM="/checkpoint/abaevski/models/libri_lms/translm2/checkpoint_best.pt"
#BEAM=50
#PREFIX="${NAME}.translm"
#CONSTRAINT="volta32gb"
#
#
#echo python decode_automl.py \
#  -d $DATA \
#  --gen-subset dev_other \
#  --log-dir ${LOGDIR}/${DECODER} \
#  -l $LEXICON \
#  -e $LOGDIR/emissions_dev_other.npy \
#  -m $MAXTOKS \
#  --decoder $DECODER \
#  --lm $LM \
#  --beam $BEAM \
#  --remove-bpe letter \
#  --prefix $PREFIX \
#  -g 4 -j 8 --num-runs 128 \
#  --partition dev,learnfair \
#  --constraint $CONSTRAINT


# python sweeps/sweep_lm_automl.py
#   -d /checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw
#   --gen-subset dev_other
#   --log-dir /checkpoint/abaevski/automl/
#   --decoder kenlm --lm /checkpoint/abaevski/data/speech/libri/4-gram.bin
#   --lexicon /checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw/lexicon_ltr2.lst
#   --targets ltr --max-tokens 4000000
#   --beam 800 --beam-threshold 80 --beam-tokens 80
#   -p small_960h_4gram
#   -g 1 -j 8 -r 128
#   --partition dev,learnfair,Wav2Vec
#   --emission /checkpoint/abaevski/emissions/small_960h.npy
#   --remove-bpe letter
