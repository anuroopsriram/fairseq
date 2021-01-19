#!/usr/bin/env bash


MODEL_10H="logs/w2v.base.8x400k.augment.ft.libris.run2/ckptlr0.0005.sourceaug1.targetaug1.augsadditive,speed.ld0.0.snr_min8.speed_std0.1.unlab.lr2e-05.mlen4.mprob0.5.dodefault.lab.10h"
MODEL_100H="logs/w2v.base.8x400k.augment.ft.libris.run2/ckptlr0.0005.sourceaug1.targetaug1.augsadditive,speed.ld0.0.snr_min8.speed_std0.1.unlab.lr2e-05.mlen4.mprob0.5.dodefault.lab.100h"
MODEL_960H="logs/w2v.base.8x400k.augment.ft.libris.run2/ckptlr0.0005.sourceaug1.targetaug1.augsadditive,speed.ld0.0.snr_min8.speed_std0.1.unlab.lr2e-05.mlen4.mprob0.5.dodefault.lab.960h"



#MODEL_10H="logs/w2v.base.8x400k.augment.ft.libris.run3/ckptlr0.0005.sourceaug1.targetaug1.augsadditive,speed.ld0.0.snr_min8.speed_std0.1.unlab.lr2e-05.mlen4.mprob0.5.dodefault.kenlm.lab.10h"
#MODEL_100H="logs/w2v.base.8x400k.augment.ft.libris.run3/ckptlr0.0005.sourceaug1.targetaug1.augsadditive,speed.ld0.0.snr_min8.speed_std0.1.unlab.lr2e-05.mlen4.mprob0.5.dodefault.kenlm.lab.100h"
#MODEL_960H="logs/w2v.base.8x400k.augment.ft.libris.run3/ckptlr0.0005.sourceaug1.targetaug1.augsadditive,speed.ld0.0.snr_min8.speed_std0.1.unlab.lr2e-05.mlen4.mprob0.5.dodefault.kenlm.lab.960h"


#python evaluate_model.py emit $MODEL_10H --splits devother
#python evaluate_model.py viterbi $MODEL_10H --splits devother
#python evaluate_model.py automl $MODEL_10H --lm kenlm --wrdsc_min -10 --wrdsc_max 0


#python evaluate_model.py emit $MODEL_100H --splits devother
#python evaluate_model.py viterbi $MODEL_100H --splits devother
#python evaluate_model.py automl $MODEL_100H --lm kenlm --wrdsc_min -10 --wrdsc_max 0


#python evaluate_model.py emit $MODEL_960H --splits devother
#python evaluate_model.py viterbi $MODEL_960H --splits devother
#python evaluate_model.py automl $MODEL_960H --lm kenlm --wrdsc_min -10 --wrdsc_max 0



#python evaluate_model.py emit $MODEL_10H --splits devother
#python evaluate_model.py viterbi $MODEL_10H --splits devother
#python evaluate_model.py automl $MODEL_10H --lm kenlm --wrdsc_min -10 --wrdsc_max 0

#python evaluate_model.py emit $MODEL_100H --splits devother
#python evaluate_model.py viterbi $MODEL_100H --splits devother
#python evaluate_model.py automl $MODEL_100H --lm kenlm --wrdsc_min -10 --wrdsc_max 0

#python evaluate_model.py emit $MODEL_960H --splits devother
#python evaluate_model.py viterbi $MODEL_960H --splits devother
#python evaluate_model.py automl $MODEL_960H --lm kenlm --wrdsc_min -10 --wrdsc_max 0





#python evaluate_model.py viterbi $MODEL_10H --splits devother0.05
#python evaluate_model.py emit $MODEL_10H --splits devother0.05
python evaluate_model.py infer $MODEL_10H --splits devother0.05 --lmwt 2 --wrdsc -3





#
#python -u examples/speech_recognition/infer.py {DATA} --gen-subset dev_other --labels ltr --lexicon {LEXICON} \
#    --path {MODEL} --dump-emissions {EMISSIONS} --results-path {RESULTS} --beam 1 --beam-size-token 100 --beam-threshold 100 \
#    --criterion ctc --normalize --w2l-decoder viterbi
#
#
#python -u decode_automl.py -d {DATA} --gen-subset dev_other --prefix dev_other -l {LEXICON} --log-dir {OUTPUT_DIR} -e {EMISSIONS} \
#    --decoder kenlm --lm {LM} --beam 500 --remove-bpe letter -g 1 -j 8 --num-runs 128 --partition dev,learnfair






#MODEL100H="logs/w2v.base.8x400k.augment.ft.libris/ckptlr0.0005.sourceaug1.targetaug1.augsadditive,speed.ld0.0.snr_min8.speed_std0.1.unlab.lr2e-05.mlen4.mprob0.5.dodefault.lab.100h"
#MODEL960H="logs/w2v.base.8x400k.augment.ft.libris/ckptlr0.0005.sourceaug1.targetaug1.augsadditive,speed.ld0.0.snr_min8.speed_std0.1.unlab.lr2e-05.mlen4.mprob0.5.dodefault.lab.960h"



#python evaluate_model.py emit $MODEL100H --splits devother0.05
#python evaluate_model.py viterbi $MODEL100H --splits devother0.05
#python evaluate_model.py infer $MODEL100H --splits devother0.05 --lmwt 3.70955916 --wrdsc -3.65905936
#python evaluate_model.py automl $MODEL100H --lm kenlm --splits devother0.05
#python evaluate_model.py automl $MODEL100H --lm kenlm --splits devother0.05 --lmwt_max 15 --wrdsc_min -15 --wrdsc_max 0
#python evaluate_model.py automl $MODEL100H --lm kenlm --local --splits devother0.05 --seed 10


#python evaluate_model.py emit $MODEL100H --splits devother0.5
#python evaluate_model.py viterbi $MODEL100H --splits devother0.5
#python evaluate_model.py infer $MODEL100H --splits devother0.5 --lmwt 4.59800437974 --wrdsc -6.4018640727573




#python evaluate_model.py emit $MODEL100H
#python evaluate_model.py automl $MODEL100H --lm kenlm
#python evaluate_model.py automl $MODEL100H --lm fairseqlm



#python evaluate_model.py emit $MODEL960H
#python evaluate_model.py automl $MODEL960H --lm kenlm
#python evaluate_model.py automl $MODEL960H --lm fairseqlm



#python evaluate_model.py viterbi $MODEL960H



#python evaluate_model.py infer $MODEL100H --lmwt 0 --wrdsc 0
#python evaluate_model.py automl $MODEL100H --lm kenlm
#python evaluate_model.py automl $MODEL100H --lm kenlm --splits devother0.5


#python evaluate_model.py infer $MODEL100H --lmwt 2.0876633607855437 --wrdsc -3.525270920220459

#python evaluate_model.py emit $MODEL100H --splits devother0.5
#python evaluate_model.py automl $MODEL100H --lm kenlm --splits devother0.5 --local

#python evaluate_model.py emit $MODEL100H --splits devother0.05
#python evaluate_model.py automl $MODEL100H --lm kenlm --splits devother0.05 --local


#python evaluate_model.py emit $MODEL960H
#python evaluate_model.py automl $MODEL960H --lm kenlm

