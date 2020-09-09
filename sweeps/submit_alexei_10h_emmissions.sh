#!/bin/bash

# fairseq lms

bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions_gpu.sh \
/checkpoint/abaevski/asr/spb_abl/2x2_8n_pen0_0_0.1_10_rl.fp16.u20000.savg.ft_bert.ltr.m_static.mstd0.mask10.mprob0.75.mc_normal.mcstd32.maskc64.mcprob0.75.fn0.0.nt_gaussian.fgm0.0.ffu10000.lr4e-05.wu5000.fd0.0.drop0.0.ad0.1.atd0.0.ms3200000.sd2337.uf4.ngpu2/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw speech_pretraining 4000000 10h_gcnn_1 /checkpoint/abaevski/emissions/spb_8n_abl.npy ltr /checkpoint/abaevski/models/libri_lms/convlm/word_gcnn_14B_antares.pt

bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions_gpu.sh \
/checkpoint/abaevski/asr/spb_abl/2x2_8n_pen0_0_0.1_10_rl.fp16.u20000.savg.ft_bert.ltr.m_static.mstd0.mask10.mprob0.75.mc_normal.mcstd32.maskc64.mcprob0.75.fn0.0.nt_gaussian.fgm0.0.ffu10000.lr4e-05.wu5000.fd0.0.drop0.0.ad0.1.atd0.0.ms3200000.sd2337.uf4.ngpu2/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw speech_pretraining 4000000 10h_translm_1 /checkpoint/abaevski/emissions/spb_8n_abl.npy ltr /checkpoint/abaevski/models/libri_lms/translm/word_transformer_adaptive_inputs_large_vineel.pt


bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions_gpu.sh \
/checkpoint/abaevski/asr/spb_abl/2x2_8n_pen0_0_0.1_10_rl.fp16.u20000.savg.ft_bert.ltr.m_static.mstd0.mask10.mprob0.75.mc_normal.mcstd32.maskc64.mcprob0.75.fn0.0.nt_gaussian.fgm0.0.ffu10000.lr4e-05.wu5000.fd0.0.drop0.0.ad0.1.atd0.0.ms3200000.sd2337.uf4.ngpu2/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw speech_pretraining 4000000 10h_gcnn_2 /checkpoint/abaevski/emissions/spb_8n_abl.npy ltr /checkpoint/abaevski/models/libri_lms/convlm2/checkpoint_best.pt

bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions_gpu.sh \
/checkpoint/abaevski/asr/spb_abl/2x2_8n_pen0_0_0.1_10_rl.fp16.u20000.savg.ft_bert.ltr.m_static.mstd0.mask10.mprob0.75.mc_normal.mcstd32.maskc64.mcprob0.75.fn0.0.nt_gaussian.fgm0.0.ffu10000.lr4e-05.wu5000.fd0.0.drop0.0.ad0.1.atd0.0.ms3200000.sd2337.uf4.ngpu2/checkpoint_best.pt \
/checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw speech_pretraining 4000000 10h_translm_2 /checkpoint/abaevski/emissions/spb_8n_abl.npy ltr /checkpoint/abaevski/models/libri_lms/translm2/checkpoint_best.pt

#960 bpe

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/spb_swp_960h_2/2x2_8n_960h_8k.fp16.8k.u320000.savg.ft_bert.m_static.mstd0.mask10.mprob0.5.mc_normal.mcstd32.maskc64.mcprob0.25.fn0.0.nt_gaussian.fgm0.0.ffu10000.lr3e-05.wu5000.fd0.0.drop0.0.ad0.1.atd0.0.ms3200000.sd2337.uf4.ngpu2/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw speech_pretraining 4000000 8k_swp_do /checkpoint/abaevski/emissions/8k_bpe_dev_other.npy 8k
#
#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/spb_swp_960h_2/2x2_8n_960h_16k.fp16.16k.u320000.savg.ft_bert.m_static.mstd0.mask10.mprob0.5.mc_normal.mcstd32.maskc64.mcprob0.25.fn0.0.nt_gaussian.fgm0.0.ffu10000.lr3e-05.wu5000.fd0.0.drop0.0.ad0.1.atd0.0.ms3200000.sd3337.uf4.ngpu2/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw speech_pretraining 4000000 16k_swp_do /checkpoint/abaevski/emissions/16k_bpe_devother.npy 16k

# speechbert

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/spb_abl/2x2_8n_pen0_0_0.1_10_rl.fp16.u20000.savg.ft_bert.ltr.m_static.mstd0.mask10.mprob0.75.mc_normal.mcstd32.maskc64.mcprob0.75.fn0.0.nt_gaussian.fgm0.0.ffu10000.lr4e-05.wu5000.fd0.0.drop0.0.ad0.1.atd0.0.ms3200000.sd2337.uf4.ngpu2/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 4000000 spb_8n_main /checkpoint/abaevski/emissions/spb_8n_abl2.npy
#
#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/spb_abl/2x2_8n_noqtz.fp16.u20000.savg.ft_bert.ltr.m_static.mstd0.mask10.mprob0.75.mc_normal.mcstd32.maskc64.mcprob0.75.fn0.0.nt_gaussian.fgm0.0.ffu10000.lr3e-05.wu5000.fd0.0.drop0.0.ad0.1.atd0.0.ms3200000.sd3337.uf4.ngpu2/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 4000000 spb_8n_noqtz /checkpoint/abaevski/emissions/spb_8n_noqtz.npy
#
#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/spb_abl/2x2_big_best.fp16.u20000.savg.ft_bert.ltr.m_static.mstd0.mask10.mprob0.75.mc_normal.mcstd32.maskc64.mcprob0.75.fn0.0.nt_gaussian.fgm0.0.ffu10000.lr3e-05.wu5000.fd0.0.drop0.0.ad0.1.atd0.0.ms1280000.sd3337.uf5.ngpu4/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 4000000 spb_16n_big /checkpoint/abaevski/emissions/spb_16n_big.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_raw_250k_q_ctc4/big250k_dr03_bsz.fp16.u60000.ft_bert.ltr.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcstd64.maskc64.mcprob0.5.fn0.1.nt_gaussian.fgm0.1.ffu10000.lr1e-05.warmup0.hld35000.dec25000.frs0.05.fd0.0.drop0.0.ad0.0.atd0.0.ms640000.sd2.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 4000000 w2v_big2_fp16 /checkpoint/abaevski/emissions/spb_big_fp16.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_raw_250k_q_ctc4/big250k_dr03_bsz.u60000.ft_bert.ltr.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcstd64.maskc64.mcprob0.5.fn0.1.nt_gaussian.fgm0.1.ffu10000.lr1e-05.warmup0.hld35000.dec25000.frs0.05.fd0.0.drop0.0.ad0.0.atd0.0.ms640000.sd2.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 4000000 w2v_big2_fp32 /checkpoint/abaevski/emissions/spb_big_fp32.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_raw_250k_q_ctc/16n_fgm.u50000.ft_bert.ltr.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcstd64.maskc64.mcprob0.25.lr2e-05.warmup5000.warmup16500.warmup28500.fd0.0.drop0.0.ad0.0.atd0.0.ms640000.sd2.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 2000000 w2v_raw_fgm /checkpoint/abaevski/emissions/spb_raw.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_raw_250k_q_ctc/16n_qint_qtz.u50000.ft_bert.ltr.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcstd64.maskc64.mcprob0.25.lr2e-05.warmup5000.warmup16500.warmup28500.fd0.0.drop0.0.ad0.0.atd0.0.ms640000.sd2.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 2000000 w2v_raw_qinp /checkpoint/abaevski/emissions/spb_raw_qinp.npy
#
#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_raw_250k_q_ctc/16fxn_bfr_qtz_fgm0.1.fp16.u50000.ft_bert.ltr.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcstd64.maskc64.mcprob0.25.lr2e-05.warmup5000.warmup16500.warmup28500.fd0.0.drop0.0.ad0.0.atd0.0.ms640000.sd2.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 2000000 w2v_raw_bfr /checkpoint/abaevski/emissions/spb_raw_bfr.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_raw_250k_q_ctc/16n_qtz_fp32.u50000.ft_bert.ltr.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcstd64.maskc64.mcprob0.25.lr2e-05.warmup5000.warmup16500.warmup28500.fd0.0.drop0.0.ad0.0.atd0.0.ms640000.sd2.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 2000000 w2v_raw_fp32 /checkpoint/abaevski/emissions/spb_raw_fp32.npy
#
#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_raw_250k_q_ctc/16n_74acc_205kupd.u50000.ft_bert.ltr.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcstd64.maskc64.mcprob0.25.lr2e-05.warmup5000.warmup16500.warmup28500.fd0.0.drop0.0.ad0.0.atd0.0.ms640000.sd2.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 2000000 w2v_raw_bfr1_expl /checkpoint/abaevski/emissions/spb_raw_bfr1_expl.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_raw_250k_q_ctc/qtz_qinp_diffq.fp16.u50000.ft_bert.ltr.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcstd64.maskc64.mcprob0.25.lr2e-05.warmup5000.warmup16500.warmup28500.fd0.0.drop0.0.ad0.0.atd0.0.ms640000.sd2.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 2000000 w2v_raw_qinp_diff /checkpoint/abaevski/emissions/spb_raw_qinp_diff.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_raw_250k_q_ctc/16n_bfrall_noqtz.fp16.u50000.ft_bert.ltr.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcstd64.maskc64.mcprob0.25.lr2e-05.warmup5000.warmup16500.warmup28500.fd0.0.drop0.0.ad0.0.atd0.0.ms640000.sd2.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 2000000 w2v_raw_bfrall_noqtz /checkpoint/abaevski/emissions/spb_raw_bfrall_noqtz.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_raw_250k_q_ctc/16n_bfr_all_qtz.fp16.u50000.ft_bert.ltr.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcstd64.maskc64.mcprob0.25.lr2e-05.warmup5000.warmup16500.warmup28500.fd0.0.drop0.0.ad0.0.atd0.0.ms640000.sd2.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 2000000 w2v_raw_bfrall /checkpoint/abaevski/emissions/spb_raw_bfrall.npy

# librivox

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/spb_ctc_10h_librivox/spb_10h_1.19m.fp16.u50000.ft.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.warmup5000.hold16500.decay28500.ffu0.fd0.0.drop0.0.ad0.0.attn_drop0.0.maxtok3072.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/quantized_librivox_ns/ translation 8000 wav2vec_q_10h_librivox /checkpoint/abaevski/emissions/librivox_base_10h.npy

# c wav2vec

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_10m/w2v_scp_10m.fp16.u20000.ft_bert.scp.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.warmup1250.hold6600.decay12150.ffu0.fd0.0.drop0.0.ad0.1.attn_drop0.1.maxtok3072.seed3.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 8000 wav2vec_c_10m_scp /checkpoint/abaevski/emissions/wav2vec_c_10m_spc.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_1h/w2v_1h_c.fp16.u20000.ft_bert.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.warmup1250.hold6600.decay12150.ffu0.fd0.0.drop0.0.ad0.1.attn_drop0.1.maxtok3072.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 2000000 wav2vec_c_1h /checkpoint/abaevski/emissions/wav2vec_c_1h.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_1h/w2v_scp_1h.fp16.u20000.ft_bert.scp.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.warmup1250.hold6600.decay12150.ffu0.fd0.0.drop0.0.ad0.2.attn_drop0.1.maxtok3072.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 8000 wav2vec_c_1h_scp /checkpoint/abaevski/emissions/wav2vec_c_1h_spc.npy
#
#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_10h/w2v_10h.fp16.u50000.ft_bert.scp.ltr.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcstd64.maskc64.mcprob0.25.zero.lr2e-05.warmup5000.warmup16500.warmup28500.fd0.0.drop0.0.ad0.0.atd0.0.maxtok3072.sd2.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 8000 wav2vec_c_10h_scp /checkpoint/abaevski/emissions/wav2vec_c_10h_spc.npy
#
#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_100h/w2v_scp_100h_c.fp16.u160000.ft_bert.scp.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.wu8000.hold52800.decay91200.fd0.0.drop0.0.ad0.0.attn_drop0.0.maxtok6144.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 8000 wav2vec_c_100h_scp /checkpoint/abaevski/emissions/wav2vec_c_100h_spc.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_10h/w2v_10h_c.fp16.u50000.ft_bert.ltr.m_normal.mstd10.mask20.mprob0.75.mc_normal.mcstd64.maskc64.mcprob0.25.zero.lr2e-05.warmup5000.warmup16500.warmup28500.fd0.0.drop0.0.ad0.0.atd0.0.ms640000.sd2.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 2500000 wav2vec_c_10h /checkpoint/abaevski/emissions/wav2vec_c_10h.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_100h/w2v_100h_c.fp16.u160000.ft_bert.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.wu8000.hold52800.decay91200.fd0.0.drop0.0.ad0.0.attn_drop0.0.maxtok640000.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 2500000 wav2vec_c_100h_2 /checkpoint/abaevski/emissions/wav2vec_c_100h.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_100h/w2v_100h_c.fp16.u160000.ft_bert.scratch.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.wu8000.hold52800.decay91200.fd0.0.drop0.0.ad0.0.attn_drop0.0.maxtok1280000.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 2000000 wav2vec_c_100h_scratch /checkpoint/abaevski/emissions/wav2vec_c_100h_scratch.npy

# c logmel

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_10m/logmel_10m_c.fp16.u20000.ft_bert.scp.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.warmup1250.hold6600.decay12150.ffu0.fd0.0.drop0.0.ad0.1.attn_drop0.1.maxtok3072.seed4.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/logmel/raw speech_pretraining 8000 logmel_c_10m_2 /checkpoint/abaevski/emissions/logmel_c_10m.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_1h/logmel_1h_c.fp16.u20000.ft_bert.scp.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.warmup1250.hold6600.decay12150.ffu0.fd0.0.drop0.0.ad0.1.attn_drop0.1.maxtok3072.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/logmel/raw speech_pretraining 8000 logmel_c_1h /checkpoint/abaevski/emissions/logmel_c_1h.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_1h/logmel_1h_c_2.fp16.u20000.ft_bert.scp.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.warmup1250.hold6600.decay12150.ffu0.fd0.0.drop0.0.ad0.1.attn_drop0.1.maxtok3072.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/logmel/raw speech_pretraining 8000 logmel_c_1h_2_new /checkpoint/abaevski/emissions/logmel_c_1h_2.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_10h/logmel_ifm.fp16.u50000.ft_bert.scp.ltr.m_normal.mstd10.mask20.mprob0.75.mc_normal.mcstd64.maskc64.mcprob0.25.zero.lr2e-05.warmup5000.warmup16500.warmup28500.fd0.0.drop0.0.ad0.0.atd0.0.maxtok3072.sd2.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/logmel/raw speech_pretraining 8000 logmel_c_10h /checkpoint/abaevski/emissions/logmel_c_10h.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_100h/logmel_100h_c_2.fp16.u160000.ft_bert.scp.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.wu8000.hold52800.decay91200.fd0.0.drop0.0.ad0.0.attn_drop0.0.maxtok6144.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/logmel/raw speech_pretraining 8000 logmel_c_100h /checkpoint/abaevski/emissions/logmel_c_100h.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_100h/logmel_100h_c.fp16.u160000.ft_bert.scratch.scp.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.wu8000.hold52800.decay91200.fd0.0.drop0.0.ad0.0.attn_drop0.0.maxtok6144.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/logmel/raw speech_pretraining 8000 logmel_c_100h_scratch /checkpoint/abaevski/emissions/logmel_c_100h_scratch.npy

# c mfcc

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_10m/mfcc_10m_c.fp16.u20000.ft_bert.scp.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.warmup1250.hold6600.decay12150.ffu0.fd0.0.drop0.0.ad0.1.attn_drop0.1.maxtok3072.seed1.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/mfcc/raw speech_pretraining 8000 mfcc_c_10m_2 /checkpoint/abaevski/emissions/mfcc_c_10m.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_1h/mfcc_1h_c.fp16.u20000.ft_bert.scp.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.warmup1250.hold6600.decay12150.ffu0.fd0.0.drop0.0.ad0.1.attn_drop0.1.maxtok3072.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/mfcc/raw speech_pretraining 8000 mfcc_c_1h /checkpoint/abaevski/emissions/mfcc_c_1h.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_10h/mfcc_ifm.fp16.u50000.ft_bert.scp.ltr.m_normal.mstd10.mask20.mprob0.75.mc_normal.mcstd64.maskc64.mcprob0.25.zero.lr2e-05.warmup5000.warmup16500.warmup28500.fd0.0.drop0.0.ad0.0.atd0.0.maxtok3072.sd2.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/mfcc/raw speech_pretraining 8000 mfcc_c_10h /checkpoint/abaevski/emissions/mfcc_c_10h.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_100h/mfcc_100h_c.fp16.u160000.ft_bert.scp.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.wu8000.hold52800.decay91200.fd0.0.drop0.0.ad0.0.attn_drop0.0.maxtok6144.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/mfcc/raw speech_pretraining 8000 mfcc_c_100h /checkpoint/abaevski/emissions/mfcc_c_100h.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_100h/mfcc_100h_c.fp16.u160000.ft_bert.scratch.scp.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.wu8000.hold52800.decay91200.fd0.0.drop0.0.ad0.0.attn_drop0.0.maxtok6144.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/mfcc/raw speech_pretraining 8000 mfcc_c_100h_scratch /checkpoint/abaevski/emissions/mfcc_c_100h_scratch.npy

# q logmel

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_10m/logmel_10m_qr.fp16.u20000.ft.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.warmup1250.hold6600.decay12150.ffu0.fd0.0.drop0.0.ad0.1.attn_drop0.1.maxtok3072.seed4.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/logmel/quantized translation 8000 logmel_q_10m /checkpoint/abaevski/emissions/logmel_q_10m.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/spb_ctc_1h_tune_conv/spb_1h_logmel.fp16.u20000.ft.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.warmup1250.hold6600.decay12150.ffu0.fd0.0.drop0.0.ad0.1.attn_drop0.1.maxtok3072.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/logmel/quantized translation 8000 logmel_q_1h_conv /checkpoint/abaevski/emissions/logmel_q_1h_conv.npy
#
#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/spb_ctc_10h_tune_conv/spb_10h_logmel.fp16.u50000.ft.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.warmup5000.hold16500.decay28500.ffu0.fd0.0.drop0.0.ad0.0.attn_drop0.0.maxtok3072.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/logmel/quantized translation 8000 logmel_q_10h_conv /checkpoint/abaevski/emissions/logmel_q_10h_conv.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/spb_ctc_100h_tune_conv/spb_100h_logmel.fp16.u320000.ft.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.wu16000.hold105600.decay198400.fd0.0.drop0.0.ad0.0.attn_drop0.0.maxtok6144.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/logmel/quantized translation 8000 logmel_q_100h_conv /checkpoint/abaevski/emissions/logmel_q_100h_conv.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/spb_ctc_100h_tune_conv/spb_100h_logmel_scratch.fp16.u320000.ft.scratch.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.wu16000.hold105600.decay198400.fd0.0.drop0.0.ad0.0.attn_drop0.0.maxtok6144.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/logmel/quantized translation 8000 logmel_q_100h_conv_scratch /checkpoint/abaevski/emissions/logmel_q_100h_conv_scratch2.npy

# q mfcc

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_10m/mfcc_10m_q.fp16.u20000.ft.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.warmup1250.hold6600.decay12150.ffu0.fd0.0.drop0.0.ad0.1.attn_drop0.1.maxtok3072.seed3.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/mfcc/quantized translation 8000 mfcc_q_10m /checkpoint/abaevski/emissions/mfcc_q_10m.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/spb_ctc_1h_tune_conv/spb_1h_mfcc.fp16.u20000.ft.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.warmup1250.hold6600.decay12150.ffu0.fd0.0.drop0.0.ad0.1.attn_drop0.1.maxtok3072.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/mfcc/quantized translation 8000 mfcc_q_1h_conv /checkpoint/abaevski/emissions/mfcc_q_1h_conv.npy
#
#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/spb_ctc_10h_tune_conv/spb_10h_mfcc.fp16.u50000.ft.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.warmup5000.hold16500.decay28500.ffu0.fd0.0.drop0.0.ad0.0.attn_drop0.0.maxtok3072.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/mfcc/quantized translation 8000 mfcc_q_10h_conv /checkpoint/abaevski/emissions/mfcc_q_10h_conv.npy
#
#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/spb_ctc_100h_tune_conv/spb_100h_mfcc.fp16.u320000.ft.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.wu16000.hold105600.decay198400.fd0.0.drop0.0.ad0.0.attn_drop0.0.maxtok6144.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/mfcc/quantized translation 8000 mfcc_q_100h_conv /checkpoint/abaevski/emissions/mfcc_q_100h_conv.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/spb_ctc_100h_tune_conv/spb_100h_mfcc.fp16.u160000.ft.scratch.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.wu8000.hold52800.decay91200.fd0.0.drop0.0.ad0.0.attn_drop0.0.maxtok6144.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/mfcc/quantized translation 8000 mfcc_q_100h_conv_scratch /checkpoint/abaevski/emissions/mfcc_q_100h_conv_scratch.npy

# quant wav2vec

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_10m/w2v_10m_q.fp16.u20000.ft.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.warmup1250.hold6600.decay12150.ffu0.fd0.0.drop0.0.ad0.1.attn_drop0.1.maxtok3072.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/quantized translation 8000 wav2vec_q_10m /checkpoint/abaevski/emissions/w2v_q_10m.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/spb_ctc_1h_tune_conv/spb_1h.fp16.u20000.ft.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.warmup1250.hold6600.decay12150.ffu0.fd0.2.drop0.0.ad0.1.attn_drop0.0.maxtok3072.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/quantized translation 8000 wav2vec_q_1h_conv /checkpoint/abaevski/emissions/wav2vec_q_1h_conv.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/spb_ctc_10h_tune_conv/spb_10h_new_mstd.fp16.u50000.ft.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.warmup5000.hold16500.decay28500.ffu0.fd0.0.drop0.0.ad0.0.attn_drop0.0.maxtok3072.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/quantized translation 8000 wav2vec_q_10h_conv /checkpoint/abaevski/emissions/wav2vec_q_10h_conv.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/spb_ctc_100h_tune_conv/spb_100h.fp16.u160000.ft.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.wu8000.hold52800.decay91200.fd0.0.drop0.0.ad0.0.attn_drop0.0.maxtok6144.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/quantized translation 8000 wav2vec_q_100h_conv /checkpoint/abaevski/emissions/wav2vec_q_100h_conv.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/spb_ctc_100h_tune_conv/spb_100h_scratch.fp16.u320000.ft.scratch.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.wu16000.hold105600.decay198400.fd0.0.drop0.0.ad0.0.attn_drop0.0.maxtok3072.seed2.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/quantized translation 8000 wav2vec_q_100h_conv_scratch /checkpoint/abaevski/emissions/w2v_q_100h_conv_scratch.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/spb_ctc_960h_tune_conv/spb_96h.fp16.u1200000.ft.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.wu16000.hold400000.decay784000.fd0.0.drop0.0.ad0.0.attn_drop0.0.maxtok6144.seed2.uf1.ngpu8/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/quantized translation 8000 wav2vec_q_960h_conv /checkpoint/abaevski/emissions/w2v_q_960h_conv.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/spb_ctc_960h_tune_conv/spb_960h_scratch.fp16.u1200000.ft.scratch.ltr.mask.m_normal.mstd0.mask20.mprob0.75.mc_normal.mcst64.mskc64.mcp0.25.lr2e-05.wu16000.hold400000.decay784000.fd0.0.drop0.0.ad0.0.attn_drop0.0.maxtok6144.seed2.uf1.ngpu8/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/quantized translation 8000 wav2vec_q_960h_scratch /checkpoint/abaevski/emissions/w2v_q_960h_scratch.npy

## old

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_1h_masks_q/wav2vec_q_1h.fp16.u20000.ft_bert.ltr.mask.m_normal.mstd10.mask10.mprob0.5.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.warmup2000.fd0.1.drop0.1.ad0.1.attn_drop0.1.maxtok2048.seed1.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/quantized translation 8000 wav2vec_q_1h /checkpoint/abaevski/emissions/wav2vec_q_1h_mask.npy
#
#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_10h_masks_q/wav2vec_q_10h.fp16.u20000.ft_bert.ltr.mask.m_uniform.mask10.mprob0.5.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.warmup2000.fd0.1.drop0.1.ad0.1.attn_drop0.1.maxtok2048.seed1.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/quantized translation 8000 wav2vec_q_10h /checkpoint/abaevski/emissions/wav2vec_q_10h_mask.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_100h_masks_q/wav2vec_q_100h.fp16.u80000.ft_bert.ltr.mask_same.mask.m_uniform.mask10.mprob0.5.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.warmup2000.fd0.1.drop0.1.ad0.1.attn_drop0.1.maxtok2048.seed1.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/100h/wav2vec/quantized translation 8000 wav2vec_q_100h_v2 /checkpoint/abaevski/emissions/wav2vec_q_100h_mask.npy

#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_1h_masks_2/cont_wav2vec_1h.fp16.u20000.ft_bert.ltr.mask.m_normal.mstd50.mask10.mprob0.5.zero.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.1.drop0.1.ad0.1.attn_drop0.1.min_sz=16000.ms640000.seed1.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 1000000 wav2vec_c_1h /checkpoint/abaevski/emissions/wav2vec_c_1h_mask.npy
#
#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_10h_masks_2/cont_wav2vec_10h.fp16.u40000.ft_bert.ltr.mask.m_normal.mstd50.mask10.mprob0.5.zero.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.1.drop0.1.ad0.1.attn_drop0.1.min_sz=16000.ms640000.seed1.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 1000000 wav2vec_c_10h /checkpoint/abaevski/emissions/wav2vec_c_10h_mask.npy
#
#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_100h_masks_2/cont_wav2vec_100h.fp16.u80000.ft_bert.ltr.mask.m_uniform.mask10.mprob0.25.zero.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.1.drop0.1.ad0.1.attn_drop0.1.min_sz=16000.ms640000.seed1.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/100h/wav2vec/raw speech_pretraining 1000000 wav2vec_c_100h_v2 /checkpoint/abaevski/emissions/wav2vec_c_100h_mask.npy





#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_1h_ltr_final3/cont_wav2vec.fp16.u20000.ft_bert.ltr.zero.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.0.drop0.1.ad0.1.attn_drop0.1.min_sz=16000.ms1280000.seed1.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/wav2vec/raw speech_pretraining 1000000 wav2vec_c_1h /checkpoint/abaevski/emissions/wav2vec_c_1h_nostack.npy



#bash ~abaevski/fairseq-py/sweeps/submit_sweep_1h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_1h_ltr_final3/cont_logmel.fp16.u20000.ft_bert.scp.min_sz=1.ltr.zero.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.0.drop0.1.ad0.1.attn_drop0.1.ms12000.seed1.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/logmel/raw speech_pretraining 6000 logmel_c_1h /checkpoint/abaevski/emissions/logmel_c_1h.npy
#
#bash ~abaevski/fairseq-py/sweeps/submit_sweep_10h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_10h_ltr_final3/cont_logmel.fp16.u20000.ft_bert.scp.min_sz=1.ltr.zero.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.1.drop0.1.ad0.1.attn_drop0.1.ms12000.seed1.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/logmel/raw speech_pretraining 6000 logmel_c_10h /checkpoint/abaevski/emissions/logmel_c_10h.npy10h
#
#bash ~abaevski/fairseq-py/sweeps/submit_sweep_1h_emissions.shs \
#/checkpoint/abaevski/asr/speechbert_ctc_1h_ltr_final3/cont_mfcc.fp16.u20000.ft_bert.scp.min_sz=1.ltr.zero.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.2.drop0.1.ad0.1.attn_drop0.1.ms12000.seed1.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/mfcc/raw speech_pretraining 6000 mfcc_c_1h /checkpoint/abaevski/emissions/mfcc_c_1h.npy
#
#bash ~abaevski/fairseq-py/sweeps/submit_sweep_1h_emissions.sh \
#/checkpoint/abaevski/asr/speechbert_ctc_10h_ltr_final3/cont_mfcc.fp16.u20000.ft_bert.scp.min_sz=1.ltr.zero.adam.beta0.9_0.98.eps1e-08.cosine.lr0.0001.final_lr1e-06.clip0.0.warmup2000.fd0.3.drop0.1.ad0.1.attn_drop0.1.ms12000.seed1.uf1.ngpu1/checkpoint_best.pt \
#/checkpoint/abaevski/data/speech/libri/10h/mfcc/raw speech_pretraining 6000 mfcc_c_10h /checkpoint/abaevski/emissions/mfcc_c_10h.npy