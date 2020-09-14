#!/usr/bin/env bash

python -u score_evals.py --dir logs/w2v.conformer.400k.ft.4glm/dim512.enclyrs17.lr0.0005/lr2e-05.lab.960h/res/kenlm
#python -u score_evals.py --dir logs/w2v.conformer.400k.ft.4glm/dim512.enclyrs17.lr0.0005/lr2e-05.lab.960h/res/translm &

python -u score_evals.py --dir logs/w2v.conformer.400k.ft.4glm/dim512.enclyrs17.lr0.0005/lr2e-05/res/kenlm
#python -u score_evals.py --dir logs/w2v.conformer.400k.ft.4glm/dim512.enclyrs17.lr0.0005/lr2e-05.lab.10h/res/translm &

wait

#cat logs/w2v.conformer.400k.ft.4glm/dim512.enclyrs17.lr0.0005/lr2e-05.lab.960h/res/kenlm/*/dev_other/score.txt | sort -k2 -g
