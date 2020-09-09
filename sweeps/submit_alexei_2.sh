#!/bin/bash

bash ~abaevski/fairseq-py/sweeps/submit_sweep.sh \
/checkpoint/abaevski/asr/speechbert_ctc_10h_ltr_final/logmel.fp16.u20000.ft_bert.ltr.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.0.drop0.1.ad0.1.attn_drop0.1.maxtok8000.seed1.uf1.ngpu1/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/10h/logmel/quantized translation 8000 logmel_q_10h_final1

bash ~abaevski/fairseq-py/sweeps/submit_sweep.sh \
/checkpoint/abaevski/asr/speechbert_ctc_10h_ltr_final/logmel.fp16.u20000.ft_bert.ltr.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.4.drop0.1.ad0.1.attn_drop0.1.maxtok8000.seed1.uf1.ngpu1/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/10h/logmel/quantized translation 8000 logmel_q_10h_final2