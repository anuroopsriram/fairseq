
# python sweep_pretrain.py -p w2v_steps_ted -D ted450 -g 8 -n 4 -u 2 --partition priority,learnfair
# python sweep_pretrain.py -p w2v_steps_ted.ls.fsh.swbd -D ted.ls.fsh.swbd.full -g 8 -n 4 -u 2 --partition priority,learnfair --resume-failed


# python sweep_finetune.py -p w2v_steps_ted_ft -D ted10 -g 2 -n 1 \
#     --checkpoints-dir logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU200k.ufreq2.layer_norm.ngpu32 \



# python sweep_finetune.py -p w2v_steps_ted_ft -D ted10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU200k.ufreq2.layer_norm.ngpu32 
# python sweep_finetune.py -p w2v_steps_ted_ft -D ted10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU400k.ufreq2.layer_norm.ngpu32 
# python sweep_finetune.py -p w2v_steps_ted_ft -D ted10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32 
# python sweep_finetune.py -p w2v_steps_ted_ft -D ted10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU1200k.ufreq2.layer_norm.ngpu32 


# python sweep_finetune.py -p w2v_steps_ted.ls.fsh.swbd_ft -D ted10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU200k.ufreq2.layer_norm.ngpu32
# python sweep_finetune.py -p w2v_steps_ted.ls.fsh.swbd_ft -D ted10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU400k.ufreq2.layer_norm.ngpu32
python sweep_finetune.py -p w2v_steps_ted.ls.fsh.swbd_ft -D ted10 -g 8 -n 2 --partition priority,learnfair,dev \
    --checkpoints-dir logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32


# python sweep_finetune.py -p w2v_steps_ted.ls.fsh.swbd_ft_ls10 -D ls10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU200k.ufreq2.layer_norm.ngpu32
# python sweep_finetune.py -p w2v_steps_ted.ls.fsh.swbd_ft_ls10 -D ls10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU400k.ufreq2.layer_norm.ngpu32
# python sweep_finetune.py -p w2v_steps_ted.ls.fsh.swbd_ft_ls10 -D ls10 -g 8 -n 2 --partition priority,learnfair,dev \
#     --checkpoints-dir logs/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32


