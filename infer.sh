

MODELDIR=logs/w2v.conformer.400k.ft.4glm/dim512.enclyrs17.lr0.0005/lr2e-05

#python examples/speech_recognition/infer.py /checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw/ \
#    --task audio_pretraining --seed 1 --nbest 1 \
#    --path ${MODELDIR}/checkpoint_best.pt \
#    --gen-subset dev_clean --results-path ${MODELDIR}/10h_4glm \
#    --w2l-decoder kenlm --lm-model /checkpoint/abaevski/data/speech/libri/4-gram.bin \
#    --lexicon /checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw/lexicon_ltr.lst \
#    --beam 500 --beam-threshold 100 --beam-size-token 100 --lm-weight 1.0625886642133924 \
#    --word-score -2.319620860198883 --sil-weight 0 --criterion ctc  --labels ltr --max-tokens 4000000 \
#    --remove-bpe letter --normalize --shard-id 0 --num-shards 4


python examples/speech_recognition/infer.py /checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw/ \
    --task audio_pretraining --seed 1 --nbest 1 \
    --path ${MODELDIR}/checkpoint_best.pt \
    --gen-subset dev_clean --results-path ${MODELDIR}/10h_4glm \
    --w2l-decoder fairseqlm --lm-model /checkpoint/abaevski/models/libri_lms/translm2/checkpoint_best.pt \
    --lexicon /checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw/lexicon_ltr.lst \
    --beam 500 --beam-threshold 100 --beam-size-token 100 --lm-weight 1.0625886642133924 \
    --word-score -2.319620860198883 --sil-weight 0 --criterion ctc  --labels ltr --max-tokens 4000000 \
    --remove-bpe letter --normalize --shard-id 0 --num-shards 16

