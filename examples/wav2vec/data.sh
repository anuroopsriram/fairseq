#!/usr/bin/env bash

RAWDATA='/datasets01_101/librispeech/062419/'
UNLAB_TRAIN=('train-clean-100' 'train-clean-360' 'train-other-500')

LAB_TRAIN=('train-clean-100')
LAB_VAL=('dev-other')

DATAROOT='/checkpoint/anuroops/data/libris/'

## Create unlab data
#mkdir -p ${DATAROOT}/tmp/unlab
#rm -rf ${DATAROOT}/tmp/unlab/*
#for part in "${UNLAB_TRAIN[@]}"; do
#  ln -s  ${RAWDATA}/${part} ${DATAROOT}/tmp/unlab/
#done
#
#mkdir -p ${DATAROOT}/unlab
#rm -rf ${DATAROOT}/unlab/*
#python examples/wav2vec/wav2vec_manifest.py ${DATAROOT}/tmp/unlab \
#   --dest ${DATAROOT}/unlab \
#   --ext flac --valid-percent 0.01



## Create labeled data
#mkdir -p ${DATAROOT}/tmp/lab_train
#rm -rf ${DATAROOT}/tmp/lab_train/*
#for part in "${LAB_TRAIN[@]}"; do
#  ln -s  ${RAWDATA}/${part} ${DATAROOT}/tmp/lab_train/
#done
#python examples/wav2vec/wav2vec_manifest.py ${DATAROOT}/tmp/lab_train \
#   --dest ${DATAROOT}/tmp/lab_train \
#   --ext flac --valid-percent 0.
#
#mkdir -p ${DATAROOT}/tmp/lab_val
#rm -rf ${DATAROOT}/tmp/lab_val/*
#for part in "${LAB_VAL[@]}"; do
#  ln -s  ${RAWDATA}/${part} ${DATAROOT}/tmp/lab_val/
#done
#python examples/wav2vec/wav2vec_manifest.py ${DATAROOT}/tmp/lab_val \
#   --dest ${DATAROOT}/tmp/lab_val \
#   --ext flac --valid-percent 0.


split=train
python examples/wav2vec/libri_labels.py \
    ${DATAROOT}/tmp/lab_train/train.tsv \
    --output-dir ${DATAROOT}/lab --output-name $split
cp ${DATAROOT}/tmp/lab_train/train.tsv ${DATAROOT}/lab/train.tsv
split=dev_other
python examples/wav2vec/libri_labels.py \
    ${DATAROOT}/tmp/lab_val/train.tsv \
    --output-dir ${DATAROOT}/lab --output-name $split
cp ${DATAROOT}/tmp/lab_val/train.tsv ${DATAROOT}/lab/dev_other.tsv

curl https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt > ${DATAROOT}/lab/dict.ltr.txt





##### OLD ######


#
#RAWDATA='/checkpoint/anuroops/data/libris/raw/'
#PROCDATA='/checkpoint/anuroops/data/libris/proc/'
#
#LABPARTITION='train-clean-100'
#VALPARTITION='dev-other'
#PROCLABDATA='/checkpoint/anuroops/data/libris/labeled/'
##PROCSUPDATA='/checkpoint/anuroops/data/libris/procsup/'
#
#
#### Create dirs
##mkdir -p ${RAWDATA}
##ln -s /datasets01_101/librispeech/062419/train-clean-100 ${RAWDATA}
##ln -s /datasets01_101/librispeech/062419/train-clean-360 ${RAWDATA}
##ln -s /datasets01_101/librispeech/062419/train-other-500 ${RAWDATA}
#
#
#### Create unlabeled data
##mkdir -p ${PROCDATA}
##rm -rf ${PROCDATA}/*
##python examples/wav2vec/wav2vec_manifest.py ${RAWDATA} \
##   --dest ${PROCDATA} \
##   --ext flac --valid-percent 0.01
#
#
##### Create labeled data
##mkdir -p ${PROCLABDATA}/${LABPARTITION}
##rm -rf ${PROCLABDATA}/${LABPARTITION}/*
##python examples/wav2vec/wav2vec_manifest.py ${RAWDATA}/${LABPARTITION} \
##    --dest ${PROCLABDATA}/${LABPARTITION} \
##    --ext flac --valid-percent 0
#mkdir -p ${PROCLABDATA}/${VALPARTITION}
#rm -rf ${PROCLABDATA}/${VALPARTITION}/*
#python examples/wav2vec/wav2vec_manifest.py \
#    /datasets01_101/librispeech/062419/${VALPARTITION} \
#    --dest ${PROCLABDATA}/${VALPARTITION} \
#    --ext flac --valid-percent 0
#
#
##split=train
##python examples/wav2vec/libri_labels.py \
##    ${PROCLABDATA}/${LABPARTITION}/train.tsv \
##    --output-dir ${PROCLABDATA}/${LABPARTITION} --output-name $split
#split=dev_other
#python examples/wav2vec/libri_labels.py \
#    ${PROCLABDATA}/${LABPARTITION}/train.tsv \
#    --output-dir ${PROCLABDATA}/${LABPARTITION} --output-name $split
#
#
##bash examples/speech_recognition/datasets/prepare-librispeech.sh ${RAWDATA} ${PROCSUPDATA}
#
