#!/usr/bin/env bash


# MODEL_800K="logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32/w2v_steps_ted.ls.fsh.swbd_ft/w2v_steps_ted.ls.fsh.swbd_ft.norm.lr5e-05.u25000.wstep5000.hstep7500.dstep12500.lrfs0_05.ufreq1.maxtok3200000.ffu0.fgm0_0.ml10.mp0_5.mcstatic.mcl64.mco0.mcp0_5.drpl0_1.sd1337.ngpu16"
# MODEL_800K="logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32/w2v_steps_ted_ft/w2v_steps_ted_ft.norm.lr2e-05.u25000.wstep5000.hstep7500.dstep12500.lrfs0_05.ufreq1.maxtok3200000.ffu0.fgm0_0.ml10.mp0_5.mcstatic.mcl64.mco0.mcp0_5.drpl0_1.sd1337.ngpu16/"


MODEL_ALL_200K="logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU200k.ufreq2.layer_norm.ngpu32/w2v_steps_ted.ls.fsh.swbd_ft/w2v_steps_ted.ls.fsh.swbd_ft.norm.lr2e-05.u25000.wstep5000.hstep7500.dstep12500.lrfs0_05.ufreq1.maxtok3200000.ffu0.fgm0_0.ml3.mp0_25.mcstatic.mcl64.mco0.mcp0_5.drpl0_1.sd1337.ngpu16"
MODEL_ALL_400K="logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU400k.ufreq2.layer_norm.ngpu32/w2v_steps_ted.ls.fsh.swbd_ft/w2v_steps_ted.ls.fsh.swbd_ft.norm.lr2e-05.u25000.wstep5000.hstep7500.dstep12500.lrfs0_05.ufreq1.maxtok3200000.ffu0.fgm0_0.ml3.mp0_25.mcstatic.mcl64.mco0.mcp0_5.drpl0_1.sd1337.ngpu16"
MODEL_ALL_800K="logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32/w2v_steps_ted.ls.fsh.swbd_ft/w2v_steps_ted.ls.fsh.swbd_ft.norm.lr2e-05.u25000.wstep5000.hstep7500.dstep12500.lrfs0_05.ufreq1.maxtok3200000.ffu0.fgm0_0.ml3.mp0_25.mcstatic.mcl64.mco0.mcp0_5.drpl0_1.sd1337.ngpu16"

# MODEL="logs/w2v.base.mlp.augment.8x400.ft.3x80k/ckptlr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025augSrc1.0.augTgt1.0.augsadditive,speed.snr-min6_snr-max15_speed-std0.15.unlab.lr1e-05.mlen3.mprob0.5.do0.1.lab.10h"
MODEL="/checkpoint/anuroops/fairseq/wav2vec/tmp2"
# python evaluate_model.py infer $MODEL --data ls.10h --lmwt=4.110990821989723 --wrdsc=1.0900210867538078 --sil=-3.532640136788191
python evaluate_model.py infer $MODEL --data ls.10h --lmwt=3.25152235317339 --wrdsc=-1.1671643716422846 --sil=-1.8055301893133318

# python evaluate_model.py viterbi $MODEL --data ls.10h
# python evaluate_model.py emit $MODEL --data ls.10h
# python evaluate_model.py automl $MODEL --data ls.10h


# python evaluate_model.py viterbi $MODEL_ALL_800K --data ted10  #--dictdata ted10
# python evaluate_model.py viterbi $MODEL_ALL_800K --data ls10  #--dictdata ted10
# python evaluate_model.py viterbi $MODEL_ALL_800K --data ls10  #--dictdata ted10
# python evaluate_model.py viterbi $MODEL_ALL_400K --data ls10  #--dictdata ted10


# python examples/speech_recognition/infer.py data/ted_lower/ted.10h \
# --task audio_pretraining \
# --nbest 1 --path ${MODEL_800K}/checkpoint_best.pt \
# --gen-subset dev --results-path RES --w2l-decoder viterbi \
# --criterion ctc --labels ltr --max-tokens 4000000 \
# --post-process letter --lm-weight 0 --word-score 0 




# MODEL100H="logs/w2v.base.8x400k.augment.ft.libris/ckptlr0.0005.sourceaug1.targetaug1.augsadditive,speed.ld0.0.snr_min8.speed_std0.1.unlab.lr2e-05.mlen4.mprob0.5.dodefault.lab.100h"
# MODEL960H="logs/w2v.base.8x400k.augment.ft.libris/ckptlr0.0005.sourceaug1.targetaug1.augsadditive,speed.ld0.0.snr_min8.speed_std0.1.unlab.lr2e-05.mlen4.mprob0.5.dodefault.lab.960h"



# #python evaluate_model.py emit $MODEL100H --splits devother0.05
# #python evaluate_model.py viterbi $MODEL100H --splits devother0.05
# python evaluate_model.py infer $MODEL100H --splits devother0.05 --lmwt 0. --wrdsc -2



# #python evaluate_model.py emit $MODEL100H
# #python evaluate_model.py automl $MODEL100H --lm kenlm
# #python evaluate_model.py automl $MODEL100H --lm fairseqlm


# #python evaluate_model.py emit $MODEL960H
# #python evaluate_model.py automl $MODEL960H --lm kenlm
# #python evaluate_model.py automl $MODEL960H --lm fairseqlm



# #python evaluate_model.py viterbi $MODEL960H



# #python evaluate_model.py infer $MODEL100H --lmwt 0 --wrdsc
# #python evaluate_model.py automl $MODEL100H --lm kenlm
# #python evaluate_model.py automl $MODEL100H --lm kenlm --splits devother0.5


# #python evaluate_model.py infer $MODEL100H --lmwt 2.0876633607855437 --wrdsc -3.525270920220459

# #python evaluate_model.py emit $MODEL100H --splits devother0.5
# #python evaluate_model.py automl $MODEL100H --lm kenlm --splits devother0.5 --local

# #python evaluate_model.py emit $MODEL100H --splits devother0.05
# #python evaluate_model.py automl $MODEL100H --lm kenlm --splits devother0.05 --local


# #python evaluate_model.py emit $MODEL960H
# #python evaluate_model.py automl $MODEL960H --lm kenlm

