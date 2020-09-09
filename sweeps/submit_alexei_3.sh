#!/bin/bash

bash ~abaevski/fairseq-py/sweeps/submit_sweep.sh \
/checkpoint/abaevski/asr/speechbert_ctc_10h_ltr_final/w2v_only.fp16.u20000.ft_bert.min_sz=1.ltr.zero.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.1.drop0.1.ad0.1.attn_drop0.1.ms1600000.seed1.uf1.ngpu1/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 10000000 wav2vec_w_final1

bash ~abaevski/fairseq-py/sweeps/submit_sweep.sh \
/checkpoint/abaevski/asr/speechbert_ctc_10h_ltr_final/w2v_only.fp16.u20000.ft_bert.scratch.min_sz=1.ltr.zero.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.3.drop0.1.ad0.1.attn_drop0.1.ms1600000.seed1.uf1.ngpu1/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 10000000 wav2vec_w_scratch