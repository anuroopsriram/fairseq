#!/bin/bash

# quant wav2vec

bash ~abaevski/fairseq-py/sweeps/submit_sweep_100h_emissions.sh \
/checkpoint/abaevski/asr/speechbert_ctc_100h/wav2vec.fp16.u20000.ft_bert.ltr.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.warmup2000.fd0.4.drop0.1.ad0.1.attn_drop0.1.maxtok8000.seed1.uf1.ngpu1/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/10h/wav2vec/quantized translation 8000 wav2vec_q /checkpoint/abaevski/emissions/wav2vec_q_100h.npy
