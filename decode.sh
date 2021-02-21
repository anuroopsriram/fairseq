
lm="ls_4gram"
data="/checkpoint/anuroops/data/libris/lab.960h"
# beam=50
beam=500
subset="dev_other"


# MLP + Aug
# exp_dir="/checkpoint/anuroops/fairseq/wav2vec/w2v.base.mlp.augment.8x400.ft.3x80k/ckptlr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025augSrc1.0.augTgt1.0.augsadditive,speed.snr-min6_snr-max15_speed-std0.15.unlab.lr1e-05.mlen3.mprob0.5.do0.1.lab.10h"
# exp_dir="/checkpoint/anuroops/fairseq/wav2vec/tmp2"
exp_dir="/checkpoint/anuroops/fairseq/wav2vec/w2v.base.8x400.baseline"


python examples/speech_recognition/hydra/infer.py --multirun \
    --config-dir=examples/speech_recognition/hydra/conf/ \
    --config-name=infer_kenlm \
    hydra/launcher=submitit_slurm \
    hydra/sweeper=ax \
    +run=slurm1 \
    +lm=$lm \
    +ax_sweep=ngram1 \
    task=audio_pretraining \
    task.data=$data \
    task.labels=ltr \
    decoding.decoder.beam=$beam \
    decoding.exp_dir=$exp_dir \
    decoding.write_sentences=false \
    decoding.unique_wer_file=true \
    dataset.gen_subset=$subset \
    dataset.max_tokens=1100000 \
    hydra.sweeper.ax_config.max_trials=1 \
    common_eval.path=${exp_dir}/checkpoint_best.pt


# python examples/speech_recognition/hydra/infer.py \
#     --config-dir=examples/speech_recognition/hydra/conf/ \
#     --config-name=infer_kenlm \
#     hydra/launcher=submitit_slurm \
#     +run=slurm1 \
#     +lm=$lm \
#     task=audio_pretraining \
#     task.data=$data \
#     task.labels=ltr \
#     decoding.decoder.beam=$beam \
#     decoding.exp_dir=$exp_dir \
#     decoding.write_sentences=false \
#     decoding.unique_wer_file=true \
#     decoding.decoder.lmweight=4.110990821989723 \
#     decoding.decoder.wordscore=1.0900210867538078 \
#     decoding.decoder.silweight=-3.532640136788191 \
#     dataset.gen_subset=$subset \
#     dataset.max_tokens=1100000 \
#     common_eval.path=${exp_dir}/checkpoint_best.pt

