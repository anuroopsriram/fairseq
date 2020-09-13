

'''
cp=$1
data=$2
task=$3             audio_pretraining
max_toks=$4         4_000_000
prefix=$5           10h_big_250k_ld02_drp02_lr001_last
lmscore=$6          2.462532475830972
wscore=$7           -0.5885126498183038
targets=$8          ltr
num_shards=$9       4
normalize=${10}     true
lm=${11}
splits=${12}

bash sweeps/submit_eval_sharded.sh \
    /checkpoint/michaelauli/asr/w2v_big/big_250k_ld02_drp02_lr001_last.fp16.u20000.savg.nrm.ltr.m_static.mstd0.mask10.mprob0.65.ld0.1.mc_static.mcstd0.maskc64.mcprob0.5.fgm0.0.ffu10000.lr3e-05.warmup2000.hld8000.dec10000.frs0.05.fd0.0.drop0.0.ad0.1.atd0.0.ms1280000.sd2337.uf5.ngpu4/checkpoint_best.pt \
    /checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw/ \
    audio_pretraining 4000000 10h_big_250k_ld02_drp02_lr001_last 2.462532475830972 -0.5885126498183038 ltr 4 true


bash sweeps/submit_eval_sharded.sh \
    /checkpoint/abaevski/asr/spb_10h_new/prenorm_ln_stable_repr_lr.fp16.u20000.savg.ft_bert.nrm.ltr.m_static.mstd0.mask10.mprob0.75.ld0.1.mc_static.mcstd0.maskc64.mcprob0.25.fgm0.0.ffu10000.lr0.0001.warmup2000.hld8000.dec10000.frs0.05.fd0.0.drop0.0.ad0.1.atd0.0.sd2337.ms1280000.uf5.ngpu4/checkpoint_best.pt \
    /checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw/ \
    speech_pretraining 4000000 10h_librivox_1m 1.0625886642133924 -2.319620860198883 ltr 16 true \
    /checkpoint/abaevski/models/libri_lms/translm2/checkpoint_best.pt

---------

bash sweeps/run_eval_sharded.sh \
    /checkpoint/michaelauli/asr/w2v_big/big_250k_ld02_drp02_lr001_last.fp16.u20000.savg.nrm.ltr.m_static.mstd0.mask10.mprob0.65.ld0.1.mc_static.mcstd0.maskc64.mcprob0.5.fgm0.0.ffu10000.lr3e-05.warmup2000.hld8000.dec10000.frs0.05.fd0.0.drop0.0.ad0.1.atd0.0.ms1280000.sd2337.uf5.ngpu4/checkpoint_best.pt \
    /checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw/ audio_pretraining 4000000 \
    /checkpoint/anuroops/speechbert_raw_sweeps_new/10h_big_250k_ld02_drp02_lr001_last \
    dev_clean 2.462532475830972 -0.5885126498183038 0 ltr 4 true

bash sweeps/score_evals.sh \
    /checkpoint/anuroops/speechbert_raw_sweeps_new/10h_big_250k_ld02_drp02_lr001_last dev_clean 4 \
    /checkpoint/michaelauli/asr/w2v_big/big_250k_ld02_drp02_lr001_last.fp16.u20000.savg.nrm.ltr.m_static.mstd0.mask10.mprob0.65.ld0.1.mc_static.mcstd0.maskc64.mcprob0.5.fgm0.0.ffu10000.lr3e-05.warmup2000.hld8000.dec10000.frs0.05.fd0.0.drop0.0.ad0.1.atd0.0.ms1280000.sd2337.uf5.ngpu4/checkpoint_best.pt


bash sweeps/run_eval_sharded.sh \
    /checkpoint/abaevski/asr/spb_10h_new/prenorm_ln_stable_repr_lr.fp16.u20000.savg.ft_bert.nrm.ltr.m_static.mstd0.mask10.mprob0.75.ld0.1.mc_static.mcstd0.maskc64.mcprob0.25.fgm0.0.ffu10000.lr0.0001.warmup2000.hld8000.dec10000.frs0.05.fd0.0.drop0.0.ad0.1.atd0.0.sd2337.ms1280000.uf5.ngpu4/checkpoint_best.pt \
    /checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw/ \
    speech_pretraining 4000000 \
    /checkpoint/anuroops/speechbert_raw_sweeps_new/10h_librivox_1m dev_clean \
    1.0625886642133924 -2.319620860198883 0 ltr 16 true \
    /checkpoint/abaevski/models/libri_lms/translm2/checkpoint_best.pt

---------

python examples/speech_recognition/infer.py \
    /checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw/ \
    --task audio_pretraining --seed 1 --nbest 1 \
    --path /checkpoint/michaelauli/asr/w2v_big/big_250k_ld02_drp02_lr001_last.fp16.u20000.savg.nrm.ltr.m_static.mstd0.mask10.mprob0.65.ld0.1.mc_static.mcstd0.maskc64.mcprob0.5.fgm0.0.ffu10000.lr3e-05.warmup2000.hld8000.dec10000.frs0.05.fd0.0.drop0.0.ad0.1.atd0.0.ms1280000.sd2337.uf5.ngpu4/checkpoint_best.pt \
    --gen-subset dev_clean \
    --results-path /checkpoint/anuroops/speechbert_raw_sweeps_new/10h_big_250k_ld02_drp02_lr001_last \
    --w2l-decoder kenlm \
    --lm-model /checkpoint/abaevski/data/speech/libri/4-gram.bin \
    --lexicon /checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw/lexicon_ltr.lst --beam 1500 \
    --beam-threshold 100 --beam-size-token 100 --lm-weight 2.462532475830972 --word-score -0.5885126498183038 \
    --sil-weight 0 --criterion ctc  --labels ltr --max-tokens 4000000 --remove-bpe letter --normalize \
    --shard-id  --num-shards 4


SUBSET=dev_clean
DECODER=kenlm
LEXICON=/checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw/lexicon_ltr.lst
BEAM=1500
BEAM_THRESH=100

python examples/speech_recognition/infer.py {DATADIR} \
    --path {MODEL}/checkpoint_best.pt \
    --task audio_pretraining --seed 1 --nbest 1 \
    --gen-subset {SUBSET} --results-path {RESULTSDIR} \
    --w2l-decoder {DECODER} --lm-model {LM} --lexicon {LEXICON} \
    --beam {BEAM} --beam-threshold {BEAM_THRESH} --beam-size-token 100 \
    --lm-weight {LMWEIGHT} --word-score {WORDSCORE} --sil-weight {SILWEIGHT} \
    --criterion ctc --labels ltr --max-tokens 4000000 --remove-bpe letter --normalize \
    --shard-id {SHARD_ID} --num-shards {NUM_SHARDS}


python examples/speech_recognition/infer.py \
    /checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw/ \
    --task speech_pretraining --seed 1 --nbest 1 \
    --path /checkpoint/abaevski/asr/spb_10h_new/prenorm_ln_stable_repr_lr.fp16.u20000.savg.ft_bert.nrm.ltr.m_static.mstd0.mask10.mprob0.75.ld0.1.mc_static.mcstd0.maskc64.mcprob0.25.fgm0.0.ffu10000.lr0.0001.warmup2000.hld8000.dec10000.frs0.05.fd0.0.drop0.0.ad0.1.atd0.0.sd2337.ms1280000.uf5.ngpu4/checkpoint_best.pt \
    --gen-subset dev_clean --results-path /checkpoint/anuroops/speechbert_raw_sweeps_new/10h_librivox_1m \
    --w2l-decoder fairseqlm \
    --lm-model /checkpoint/abaevski/models/libri_lms/translm2/checkpoint_best.pt \
    --lexicon /checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw/lexicon_ltr.lst \
    --beam 500 --beam-threshold 100 --beam-size-token 100 --lm-weight 1.0625886642133924 \
    --word-score -2.319620860198883 --sil-weight 0 --criterion ctc  --labels ltr --max-tokens 4000000 \
    --remove-bpe letter --normalize --shard-id  --num-shards 16

DECODER=fairseqlm
BEAM=500
BEAM_THRESH=100

python examples/speech_recognition/infer.py {DATADIR} \
    --path {MODEL}/checkpoint_best.pt \
    --task audio_pretraining --seed 1 --nbest 1 \
    --gen-subset {SUBSET} --results-path {RESULTSDIR} \
    --w2l-decoder {DECODER} --lm-model {LM} --lexicon {LEXICON} \
    --beam {BEAM} --beam-threshold {BEAM_THRESH} --beam-size-token 100 \
    --lm-weight {LMWEIGHT} --word-score {WORDSCORE} --sil-weight {SILWEIGHT} \
    --criterion ctc --labels ltr --max-tokens 4000000 --remove-bpe letter --normalize \
    --shard-id {SHARD_ID} --num-shards {NUM_SHARDS}
'''

from pathlib import Path


def add_args(parser):
    parser.add_argument('--prefix', type=str, required=True)

    parser.add_argument('--script', type=Path, default='sweeps/submit_eval_sharded.sh')
    parser.add_argument('--data', type=Path, default='/checkpoint/anuroops/data/libris/lab.960h/')
    parser.add_argument('--task', type=str, default='audio_pretraining')
    parser.add_argument('--max_toks', type=int, default=4000000)
    parser.add_argument('--lmscore', type=float, default=4000000)
