#!/bin/bash

module purge
conda deactivate
source ~abaevski/start_env
conda activate ~abaevski/.conda/envs/fairseq-20190809

# quant wav2vec

# cont wav2vec
bash ~abaevski/fairseq-py/sweeps/submit_sweep.sh \
/checkpoint/abaevski/asr/speechbert_ctc_10h_ltr_final/cont_wav2vec.fp16.u20000.ft_bert.ltr.zero.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.0.drop0.1.ad0.1.attn_drop0.1.min_sz=16000.ms1280000.seed1.uf1.ngpu1/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 10000000 wav2vec_c

bash ~abaevski/fairseq-py/sweeps/submit_sweep.sh \
/checkpoint/abaevski/asr/cont_wav2vec.fp16.u20000.ft_bert.scratch.ltr.zero.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.4.drop0.1.ad0.1.attn_drop0.1.min_sz=16000.ms1280000.seed1.uf1.ngpu1/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 10000000 wav2vec_c_scratch

# quant mfcc

# cont mfcc

bash ~abaevski/fairseq-py/sweeps/submit_sweep.sh \
/checkpoint/abaevski/asr/speechbert_ctc_10h_ltr_final/cont_mfcc.fp16.u20000.ft_bert.scp.min_sz=1.ltr.zero.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.4.drop0.1.ad0.1.attn_drop0.1.ms9600.seed1.uf1.ngpu1/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/10h/mfcc/raw speech_pretraining 8000 mfcc_c

bash ~abaevski/fairseq-py/sweeps/submit_sweep.sh \
/checkpoint/abaevski/asr/speechbert_ctc_10h_ltr_final/cont_mfcc.fp16.u20000.ft_bert.scratch.scp.min_sz=1.ltr.zero.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.2.drop0.1.ad0.1.attn_drop0.1.ms9600.seed1.uf1.ngpu1/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/10h/mfcc/raw speech_pretraining 8000 mfcc_c_scratch

# quant logmel

# cont logmel

bash ~abaevski/fairseq-py/sweeps/submit_sweep.sh \
/checkpoint/abaevski/asr/speechbert_ctc_10h_ltr_final/cont_logmel.fp16.u20000.ft_bert.scp.min_sz=1.ltr.zero.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.1.drop0.1.ad0.1.attn_drop0.1.ms9600.seed1.uf1.ngpu1/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/10h/logmel/raw speech_pretraining 8000 logmel_c

bash ~abaevski/fairseq-py/sweeps/submit_sweep.sh \
/checkpoint/abaevski/asr/speechbert_ctc_10h_ltr_final/cont_logmel.fp16.u20000.ft_bert.scratch.scp.min_sz=1.ltr.zero.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.2.drop0.1.ad0.1.attn_drop0.1.ms9600.seed1.uf1.ngpu1/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/10h/logmel/raw speech_pretraining 8000 logmel_c_scratch