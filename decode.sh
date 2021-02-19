
lm=ls_4gram
data=/checkpoint/anuroops/data/libris/lab.960h
beam=50


# MLP + Aug
exp_dir=ckptlr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025augSrc1.0.augTgt1.0.augsadditive,speed.snr-min6_snr-max15_speed-std0.15.unlab.lr1e-05.mlen3.mprob0.5.do0.1.lab.10h


python examples/speech_recognition/hydra/infer.py --multirun \
    --config-dir=examples/speech_recognition/hydra/conf/ \
    --config-name=infer_kenlm \
    hydra/launcher=submitit_slurm \
    hydra/sweeper=ax \
    +run=decode_ngram \
    +lm=$lm \
    +ax_sweep=decode_ngram \
    task=audio_pretraining \
    task.data=$data \
    task.labels=ltr \
    decoding.decoder.beam=$beam \
    decoding.exp_dir=$exp_dir \
    decoding.write_sentences=false \
    decoding.unique_wer_file=true \
    dataset.gen_subset=${splits[i]} \
    dataset.max_tokens=1100000 \
    common_eval.path=${exp_dir}/checkpoint_best.pt &
