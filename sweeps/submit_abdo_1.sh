#!/bin/bash

module purge
conda deactivate
source ~abaevski/start_env
conda activate ~abaevski/.conda/envs/fairseq-20190809

bash ~abaevski/fairseq-py/sweeps/submit_sweep.sh \
/checkpoint/abaevski/asr/speechbert_ctc_10h_2/wav2vec_ltr.fp16.u20000.ft_bert.ltr.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.3.drop0.1.ad0.1.attn_drop0.1.maxtok12000.seed1.uf1.ngpu1/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/10h/wav2vec/quantized translation 8000 wav2vec_q_10h_2

bash ~abaevski/fairseq-py/sweeps/submit_sweep.sh \
/checkpoint/abaevski/asr/speechbert_ctc_10h_ltr_final/wav2vec_ltr.fp16.u20000.ft_bert.ltr.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.3.drop0.1.ad0.1.attn_drop0.1.maxtok8000.seed1.uf1.ngpu1/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/10h/wav2vec/quantized translation 8000 wav2vec_q_final1

bash ~abaevski/fairseq-py/sweeps/submit_sweep.sh \
/checkpoint/abaevski/asr/speechbert_ctc_10h_ltr_final/wav2vec_ltr.fp16.u20000.ft_bert.ltr.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.0.drop0.1.ad0.1.attn_drop0.1.maxtok8000.seed1.uf1.ngpu1/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/10h/wav2vec/quantized translation 8000 wav2vec_q_final2

bash ~abaevski/fairseq-py/sweeps/submit_sweep.sh \
/checkpoint/abaevski/asr/speechbert_ctc_10h_ltr_final/wav2vec_ltr.fp16.u20000.ft_bert.scratch.ltr.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.4.drop0.1.ad0.1.attn_drop0.1.maxtok8000.seed1.uf1.ngpu1/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/10h/wav2vec/quantized translation 8000 wav2vec_q_scratch

bash ~abaevski/fairseq-py/sweeps/submit_sweep.sh \
/checkpoint/abaevski/asr/speechbert_ctc_10h_ltr_final/logmel.fp16.u20000.ft_bert.scratch.ltr.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.4.drop0.1.ad0.1.attn_drop0.1.maxtok8000.seed1.uf1.ngpu1/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/10h/logmel/quantized translation 8000 logmel_q_10h_scratch

bash ~abaevski/fairseq-py/sweeps/submit_sweep.sh \
/checkpoint/abaevski/asr/speechbert_ctc_10h_ltr_final/cont_wav2vec.fp16.u20000.ft_bert.ltr.zero.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.2.drop0.1.ad0.1.attn_drop0.1.min_sz=16000.ms1600000.seed1.uf1.ngpu1/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 10000000 wav2vec_c_final1

bash ~abaevski/fairseq-py/sweeps/submit_sweep.sh \
/checkpoint/abaevski/asr/speechbert_ctc_10h_ltr_final/cont_wav2vec.fp16.u20000.ft_bert.ltr.zero.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.1.drop0.1.ad0.1.attn_drop0.1.min_sz=16000.ms1600000.seed1.uf1.ngpu1/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 10000000 wav2vec_c_final2