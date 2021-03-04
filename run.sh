

## PT MODELS
PT_TD="logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32"
PT_LS="logs/w2v_steps_ls/LS.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq1.layer_norm.ngpu64"
PT_SF="logs/w2v_steps_fsh.swbd/SF.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq1.layer_norm.ngpu64"
PT_TD_LS="logs/w2v_steps_td.ls/TD_LS.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq1.layer_norm.ngpu64"
PT_TD_SF="logs/w2v_steps_ted.fsh.swbd/TD_SF.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq1.lr5e-05.layer_norm.ngpu64"
PT_LS_SF="logs/w2v_steps_ls.fsh.swbd/LS_SF.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq1.layer_norm.ngpu64"
PT_TD_LS_SF="logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32"

## FT on TED-10H
FT_TD="logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32/w2v_steps_ted_ft/w2v_steps_ted_ft.norm.lr2e-05.u25000.wstep5000.hstep7500.dstep12500.lrfs0_05.ufreq1.maxtok3200000.ffu0.fgm0_0.ml5.mp0_25.mcstatic.mcl64.mco0.mcp0_5.drpl0_1.sd1337.ngpu16"
FT_LS="logs/w2v_steps_ls/LS.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq1.layer_norm.ngpu64/w2v_ls/w2v_ls.norm.lr2e-05.u25000.wstep5000.hstep7500.dstep12500.lrfs0_05.ufreq1.maxtok3200000.ffu0.fgm0_0.ml5.mp0_25.mcstatic.mcl64.mco0.mcp0_5.drpl0_1.sd1337.ngpu16"
FT_SF="logs/w2v_steps_fsh.swbd/SF.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq1.layer_norm.ngpu64/w2v_fsh.swbd_ft_ted/w2v_fsh.swbd_ft_ted.norm.lr2e-05.u25000.wstep5000.hstep7500.dstep12500.lrfs0_05.ufreq1.maxtok3200000.ffu0.fgm0_0.ml5.mp0_25.mcstatic.mcl64.mco0.mcp0_5.drpl0_1.sd1337.ngpu16"
FT_TD_LS="logs/w2v_steps_td.ls/TD_LS.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq1.layer_norm.ngpu64/w2v_ted.ls_ft_ted/w2v_ted.ls_ft_ted.norm.lr2e-05.u25000.wstep5000.hstep7500.dstep12500.lrfs0_05.ufreq1.maxtok3200000.ffu0.fgm0_0.ml5.mp0_25.mcstatic.mcl64.mco0.mcp0_5.drpl0_1.sd1337.ngpu16"
FT_TD_SF="logs/w2v_steps_ted.fsh.swbd/TD_SF.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq1.lr5e-05.layer_norm.ngpu64/w2v_ted.ls_ft_ted/w2v_ted.ls_ft_ted.norm.lr2e-05.u25000.wstep5000.hstep7500.dstep12500.lrfs0_05.ufreq1.maxtok3200000.ffu0.fgm0_0.ml5.mp0_25.mcstatic.mcl64.mco0.mcp0_5.drpl0_1.sd1337.ngpu16"
FT_LS_SF="logs/w2v_steps_ls.fsh.swbd/LS_SF.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq1.layer_norm.ngpu64/w2v_ls.fsh.swbd_ft_ted/w2v_ls.fsh.swbd_ft_ted.norm.lr2e-05.u25000.wstep5000.hstep7500.dstep12500.lrfs0_05.ufreq1.maxtok3200000.ffu0.fgm0_0.ml5.mp0_25.mcstatic.mcl64.mco0.mcp0_5.drpl0_1.sd1337.ngpu16"
FT_TD_LS_SF="logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32/w2v_steps_ted.ls.fsh.swbd_ft/w2v_steps_ted.ls.fsh.swbd_ft.norm.lr2e-05.u25000.wstep5000.hstep7500.dstep12500.lrfs0_05.ufreq1.maxtok3200000.ffu0.fgm0_0.ml3.mp0_25.mcstatic.mcl64.mco0.mcp0_5.drpl0_1.sd1337.ngpu16"

## FT on SWBD-10H
FT2_TD=""
FT2_LS=""
FT2_SF="logs/w2v_steps_fsh.swbd/SF.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq1.layer_norm.ngpu64/w2v_steps_fsh.swbd_ft_swbd10/w2v_steps_fsh.swbd_ft_swbd10.norm.lr8e-05.u25000.wstep2500.hstep0.dstep22500.lrfs0_05.ufreq1.maxtok1280000.ffu0.fgm0_0.ml10.mp0_3.mcstatic.mcl64.mco0.mcp0_5.drpl0_1.sd1337.ngpu16"
FT2_TD_LS=""
FT2_TD_SF=""
FT2_LS_SF="logs/w2v_steps_ls.fsh.swbd/LS_SF.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq1.layer_norm.ngpu64/w2v_steps_ls.fsh.swbd_ft_swbd10/w2v_steps_ls.fsh.swbd_ft_swbd10.norm.lr8e-05.u25000.wstep2500.hstep0.dstep22500.lrfs0_05.ufreq1.maxtok1280000.ffu5000.fgm0_0.ml10.mp0_3.mcstatic.mcl64.mco0.mcp0_5.drpl0_1.sd1337.ngpu24"
FT2_TD_LS_SF="logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32/w2v_steps_ted.ls.fsh.swbd_ft_swbd10/w2v_steps_ted.ls.fsh.swbd_ft_swbd10.norm.lr2e-05.u25000.wstep5000.hstep7500.dstep12500.lrfs0_05.ufreq1.maxtok3200000.ffu0.fgm0_0.ml12.mp0_25.mcstatic.mcl64.mco0.mcp0_5.drpl0_1.sd1337.ngpu16"

# Continue on LS10H
PT_CONT_LS10_100K="logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32/cont_ls_pt/cont_ls_pt.s1337.MU100k.ufreq1.lr0_0001.ngpu16"
PT_CONT_LS10_50K="logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32/cont_ls_pt/cont_ls_pt.s1337.MU50k.ufreq1.lr5e-05.ngpu16"
PT_CONT_LS10_25K="logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32/cont_ls_pt/cont_ls_pt.s1337.MU25k.ufreq1.lr5e-05.ngpu16"
# Continue on LS100H
PT_CONT_LS100_100K="logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32/cont_ls100_pt/cont_ls100_pt.s1337.MU100k.ufreq1.lr0_0001.ngpu16"
PT_CONT_LS100_50K="logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32/cont_ls100_pt/cont_ls100_pt.s1337.MU50k.ufreq1.lr5e-05.ngpu16"
PT_CONT_LS100_25K="logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32/cont_ls100_pt/cont_ls100_pt.s1337.MU25k.ufreq1.lr5e-05.ngpu16"


### CONTINUATION
# python sweep_pretrain_continue.py -p cont_ls_pt -D ls10 --checkpoints-dir $PT_TD -g 8 -n 2 --partition priority,learnfair,dev 
python sweep_pretrain_continue.py -p cont_ls100_pt -D ls100 --checkpoints-dir $PT_TD -g 8 -n 2 --partition priority,learnfair,dev 

# python sweep_finetune_ls.py -p cont_ls_pt -D ls10 -g 8 -n 3 --checkpoints-dir $PT_CONT_LS10_25K
# python sweep_finetune_ls.py -p cont_ls_pt -D ls10 -g 8 -n 3 --checkpoints-dir $PT_CONT_LS10_50K
# python sweep_finetune_ls.py -p cont_ls_pt -D ls10 -g 8 -n 3 --checkpoints-dir $PT_CONT_LS10_100K
# python sweep_finetune_ls.py -p cont_ls_pt -D ls10 -g 8 -n 3 --checkpoints-dir $PT_CONT_LS100_25K
# python sweep_finetune_ls.py -p cont_ls_pt -D ls10 -g 8 -n 3 --checkpoints-dir $PT_CONT_LS100_50K
# python sweep_finetune_ls.py -p cont_ls_pt -D ls10 -g 8 -n 3 --checkpoints-dir $PT_CONT_LS100_100K


# ### N-Gram results
# for mdl in $FT_TD $FT_LS $FT_SF $FT_TD_LS $FT_TD_SF $FT_LS_SF $FT_TD_LS_SF $FT2_SF $FT2_LS_SF $FT2_TD_LS_SF; do
#     echo $mdl
#     for split in ted_dev ls_dev_other swbd_dev_rt03 cv_dev; do
#         wer=$(grep "Word error rate" ${mdl}/decode/kenlm_ax/${split}/*/*.out | awk '{print $NF}' | sort -g | head -n 1)
#         num=$(grep "Word error rate" ${mdl}/decode/kenlm_ax/${split}/*/*.out | wc -l)
#         if [ $num -ne 0 ]; then
#             echo -e "\t$split \t$wer \t$num"
#         fi
#     done
#     echo
# done


# python evaluate_model.py viterbi $FT2_SF --data ted10
# python evaluate_model.py viterbi $FT2_SF --data ls10
# python evaluate_model.py viterbi $FT2_SF --data swbd10
# python evaluate_model.py viterbi $FT2_SF --data cv
# python evaluate_model.py viterbi $FT2_SF --data vp

# python evaluate_model.py viterbi $FT2_LS_SF --data ted10
# python evaluate_model.py viterbi $FT2_LS_SF --data ls10
# python evaluate_model.py viterbi $FT2_LS_SF --data swbd10
# python evaluate_model.py viterbi $FT2_LS_SF --data cv
# python evaluate_model.py viterbi $FT2_LS_SF --data vp

# python evaluate_model.py viterbi $FT2_TD_LS_SF --data ted10
# python evaluate_model.py viterbi $FT2_TD_LS_SF --data ls10
# python evaluate_model.py viterbi $FT2_TD_LS_SF --data swbd10
# python evaluate_model.py viterbi $FT2_TD_LS_SF --data cv
# python evaluate_model.py viterbi $FT2_TD_LS_SF --data vp

# python evaluate_model.py viterbi $FT_TD --data ted10
# python evaluate_model.py viterbi $FT_TD --data ls10
# python evaluate_model.py viterbi $FT_TD --data swbd10
# python evaluate_model.py viterbi $FT_TD --data cv
# python evaluate_model.py viterbi $FT_TD --data vp

# python evaluate_model.py viterbi $FT_SF --data ted10
# python evaluate_model.py viterbi $FT_SF --data ls10
# python evaluate_model.py viterbi $FT_SF --data swbd10
# python evaluate_model.py viterbi $FT_SF --data cv
# python evaluate_model.py viterbi $FT_SF --data vp

# python evaluate_model.py viterbi $FT_LS_SF --data ted10
# python evaluate_model.py viterbi $FT_LS_SF --data ls10
# python evaluate_model.py viterbi $FT_LS_SF --data swbd10
# python evaluate_model.py viterbi $FT_LS_SF --data cv
# python evaluate_model.py viterbi $FT_LS_SF --data vp

# python evaluate_model.py viterbi $FT_TD_LS --data ted10
# python evaluate_model.py viterbi $FT_TD_LS --data ls10
# python evaluate_model.py viterbi $FT_TD_LS --data swbd10
# python evaluate_model.py viterbi $FT_TD_LS --data cv
# python evaluate_model.py viterbi $FT_TD_LS --data vp

# python evaluate_model.py viterbi $FT_LS --data ted10
# python evaluate_model.py viterbi $FT_LS --data ls10
# python evaluate_model.py viterbi $FT_LS --data swbd10
# python evaluate_model.py viterbi $FT_LS --data cv
# python evaluate_model.py viterbi $FT_LS --data vp

# python evaluate_model.py viterbi $FT_TD_SF --data ted10
# python evaluate_model.py viterbi $FT_TD_SF --data ls10
# python evaluate_model.py viterbi $FT_TD_SF --data swbd10
# python evaluate_model.py viterbi $FT_TD_SF --data cv
# python evaluate_model.py viterbi $FT_TD_SF --data vp


# python evaluate_model.py viterbi $FT_TD_LS_SF --data ted10
# python evaluate_model.py viterbi $FT_TD_LS_SF --data ls10
# python evaluate_model.py viterbi $FT_TD_LS_SF --data swbd10
# python evaluate_model.py viterbi $FT_TD_LS_SF --data cv
# python evaluate_model.py viterbi $FT_TD_LS_SF --data vp



# python sweep_pretrain.py -p w2v_steps_ted -D ted450 -g 8 -n 4 -u 2 --partition priority,learnfair
# python sweep_pretrain.py -p w2v_steps_ted.ls.fsh.swbd -D ted.ls.fsh.swbd.full -g 8 -n 4 -u 2 --partition priority,learnfair --resume-failed

# python sweep_pretrain_large.py -p w2v_large_lv.fsh.swbd -D lv.fsh.swbd -g 8 -n 8 -u 2 --partition priority,learnfair --resume-failed


# python sweep_finetune.py -p w2v_steps_ted_ft -D ted10 -g 2 -n 1 \
#     --checkpoints-dir logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU200k.ufreq2.layer_norm.ngpu32 \


# python sweep_finetune_ted.py -p w2v_bal_natural_ft_ted -D ted10 -g 8 -n 2 --partition priority,learnfair,dev --checkpoints-dir logs/w2v_balancing/natural --last
# python sweep_finetune_ted.py -p w2v_bal_mmasr0.5_ft_ted -D ted10 -g 8 -n 2 --partition priority,learnfair,dev --checkpoints-dir logs/w2v_balancing/mmasr0.5 --last --resume-failed
# python sweep_finetune_ted.py -p w2v_bal_unif_ft_ted -D ted10 -g 8 -n 2 --partition priority,learnfair,dev --checkpoints-dir logs/w2v_balancing/unif --last
# python sweep_finetune_ted.py -p w2v_bal_xlsr0.5_ft_ted -D ted10 -g 8 -n 2 --partition priority,learnfair,dev --checkpoints-dir logs/w2v_balancing/xlsr0.5 --last
# python sweep_finetune_ted.py -p w2v_fsh.swbd_ft_ted -D ted10 -g 8 -n 2 --partition priority,learnfair,dev --checkpoints-dir logs/w2v_steps_fsh.swbd/SF.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq1.layer_norm.ngpu64
# python sweep_finetune_ted.py -p w2v_ls.fsh.swbd_ft_ted -D ted10 -g 8 -n 2 --partition priority,learnfair,dev --checkpoints-dir logs/w2v_steps_ls.fsh.swbd/LS_SF.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq1.layer_norm.ngpu64
# python sweep_finetune_ted.py -p w2v_ls -D ted10 -g 8 -n 2 --partition priority,learnfair,dev --checkpoints-dir $PT_LS

# python sweep_finetune_ted.py -p w2v_ted.ls_ft_ted -D ted10 -g 8 -n 2 --partition priority,learnfair,dev --checkpoints-dir $PT_TD_LS --last
# python sweep_finetune_ted.py -p w2v_ted.ls_ft_ted -D ted10 -g 8 -n 2 --partition priority,learnfair,dev --checkpoints-dir $PT_TD_LS --last
# python sweep_finetune_ted.py -p w2v_ted.ls_ft_ted -D ted10 -g 8 -n 2 --partition priority,learnfair,dev --checkpoints-dir $PT_TD_SF --last


# python sweep_finetune.py -p w2v_steps_ted_ft -D ted10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU200k.ufreq2.layer_norm.ngpu32 
# python sweep_finetune.py -p w2v_steps_ted_ft -D ted10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU400k.ufreq2.layer_norm.ngpu32 
# python sweep_finetune.py -p w2v_steps_ted_ft -D ted10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32 
# python sweep_finetune.py -p w2v_steps_ted_ft -D ted10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU1200k.ufreq2.layer_norm.ngpu32 


# python sweep_finetune.py -p w2v_steps_ted_ft -D ted10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU200k.ufreq2.layer_norm.ngpu32 --resume-failed
# python sweep_finetune.py -p w2v_steps_ted_ft -D ted10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU400k.ufreq2.layer_norm.ngpu32 --resume-failed
# python sweep_finetune.py -p w2v_steps_ted_ft -D ted10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32 --resume-failed


# python sweep_finetune.py -p w2v_steps_ted.ls.fsh.swbd_ft -D ted10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU200k.ufreq2.layer_norm.ngpu32
# python sweep_finetune.py -p w2v_steps_ted.ls.fsh.swbd_ft -D ted10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU400k.ufreq2.layer_norm.ngpu32
# python sweep_finetune.py -p w2v_steps_ted.ls.fsh.swbd_ft -D ted10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32 --resume-failed


# python sweep_finetune.py -p w2v_steps_ted.ls.fsh.swbd_ft_swbd10 -D swbd10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32


# python sweep_finetune.py -p w2v_steps_ted.ls.fsh.swbd_ft_ls10 -D ls10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU200k.ufreq2.layer_norm.ngpu32
# python sweep_finetune.py -p w2v_steps_ted.ls.fsh.swbd_ft_ls10 -D ls10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU400k.ufreq2.layer_norm.ngpu32
# python sweep_finetune.py -p w2v_steps_ted.ls.fsh.swbd_ft_ls10 -D ls10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32 --resume-failed


# python sweep_finetune_swbd.py -p w2v_steps_ted.ls.fsh.swbd_ft_swbd10 -D swbd10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU200k.ufreq2.layer_norm.ngpu32 --resume-failed
# python sweep_finetune_swbd.py -p w2v_steps_ted.ls.fsh.swbd_ft_swbd10 -D swbd10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU400k.ufreq2.layer_norm.ngpu32 --resume-failed
# python sweep_finetune_swbd.py -p w2v_steps_ted.ls.fsh.swbd_ft_swbd10 -D swbd10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32 --resume-failed


# python sweep_finetune_swbd.py -p w2v_steps_fsh.swbd_ft_swbd10 -D swbd10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_fsh.swbd/SF.s1337.mxsz250000.mnsz32000.maxtok1400000.MU200k.ufreq1.layer_norm.ngpu64 --resume-failed
# python sweep_finetune_swbd.py -p w2v_steps_fsh.swbd_ft_swbd10 -D swbd10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_fsh.swbd/SF.s1337.mxsz250000.mnsz32000.maxtok1400000.MU400k.ufreq1.layer_norm.ngpu64 --resume-failed
# python sweep_finetune_swbd.py -p w2v_steps_fsh.swbd_ft_swbd10 -D swbd10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_fsh.swbd/SF.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq1.layer_norm.ngpu64 --resume-failed




# python sweep_finetune_swbd.py -p w2v_steps_ted.ls.fsh.swbd_ft_swbd10 -D swbd10 -g 8 -n 3 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU200k.ufreq2.layer_norm.ngpu32
# python sweep_finetune_swbd.py -p w2v_steps_ted.ls.fsh.swbd_ft_swbd10 -D swbd10 -g 8 -n 3 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU400k.ufreq2.layer_norm.ngpu32
# python sweep_finetune_swbd.py -p w2v_steps_ted.ls.fsh.swbd_ft_swbd10 -D swbd10 -g 8 -n 3 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32



# python sweep_finetune_swbd.py -p w2v_steps_ls.fsh.swbd_ft_swbd10 -D swbd10 -g 8 -n 3 --partition priority,learnfair \
#     --checkpoints-dir logs/w2v_steps_ls.fsh.swbd/LS_SF.s1337.mxsz250000.mnsz32000.maxtok1400000.MU200k.ufreq1.layer_norm.ngpu64
# python sweep_finetune_swbd.py -p w2v_steps_ls.fsh.swbd_ft_swbd10 -D swbd10 -g 8 -n 3 --partition priority,learnfair \
#     --checkpoints-dir logs/w2v_steps_ls.fsh.swbd/LS_SF.s1337.mxsz250000.mnsz32000.maxtok1400000.MU400k.ufreq1.layer_norm.ngpu64
# python sweep_finetune_swbd.py -p w2v_steps_ls.fsh.swbd_ft_swbd10 -D swbd10 -g 8 -n 3 --partition priority,learnfair \
#     --checkpoints-dir logs/w2v_steps_ls.fsh.swbd/LS_SF.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq1.layer_norm.ngpu64


# fairseq-hydra-train \
#     distributed_training.distributed_port=$PORT \
#     task.data=/path/to/data \
#     model.w2v_path=/path/to/model.pt \
#     --config-dir /path/to/fairseq-py/examples/wav2vec/config/finetuning \
#     --config-name base_100h
