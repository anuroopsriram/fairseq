
DATASETS = {
    "ls960": {
        "train": "ls960/train",
        "val": "ls960/dev_other",
    },

    "ted450": {
        "train": "ted/ted.450h/train",
        "val": "ted/ted.450h/dev",
    },

    "ted.ls.fsh.swbd.full": {
        "train": "ted.ls.fsh.swbd/ted.ls.fsh.swbd.full/train",
        "val": "ted.ls.fsh.swbd/ted.ls.fsh.swbd.full/dev",
    },

    "lv.fsh.swbd": {
        "train": "lv.fsh.swbd/train",
        "val": "lv.fsh.swbd/valid_lv,lv.fsh.swbd/valid_fsh.swbd",
    }
}

LAB_DATASETS = {
    "ls10": {
        "train": "ls960_lower/train_10h",
        "val": "ls960_lower/dev_other",
        "steps": 25000,
    },
    "ted10": {
        "train": "ted_lower/ted.10h/train",
        "val": "ted_lower/ted.10h/dev",
        "steps": 25000,
    },
    "swbd10": {
        "train": "joint_swbd/train_swbd_10h",
        "val": "joint_swbd/swbd_dev_rt03",
        "steps": 25000,
    }
}

LM = {
    "ls": "...",
}

MODELS = {
    "ted": {
        "200K": "/checkpoint/anuroops/fairseq/robustw2v/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU200k.ufreq2.layer_norm.ngpu32/checkpoint_best.pt",
        "400K": "/checkpoint/anuroops/fairseq/robustw2v/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU400k.ufreq2.layer_norm.ngpu32/checkpoint_best.pt",
        "800K": "/checkpoint/anuroops/fairseq/robustw2v/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32/checkpoint_best.pt",
        "1200K": "/checkpoint/anuroops/fairseq/robustw2v/w2v_steps_ted/w2v_steps_ted.s1337.mxsz250000.mnsz32000.maxtok1400000.MU1200k.ufreq2.layer_norm.ngpu32/checkpoint_best.pt",
    },
    "fsh.swbd": {
        "200K": "/checkpoint/wnhsu/experiments/wav2vec2_robust_b1b43cd/SF/SF.s1337.mxsz250000.mnsz32000.maxtok1400000.MU200k.ufreq1.layer_norm.ngpu64/checkpoint_last.pt",
        "400K": "/checkpoint/wnhsu/experiments/wav2vec2_robust_b1b43cd/SF/SF.s1337.mxsz250000.mnsz32000.maxtok1400000.MU400k.ufreq1.layer_norm.ngpu64/checkpoint_last.pt",
        "800K": "/checkpoint/wnhsu/experiments/wav2vec2_robust_b1b43cd/SF/SF.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq1.layer_norm.ngpu64/checkpoint_last.pt",
        "1200K": "/checkpoint/wnhsu/experiments/wav2vec2_robust_b1b43cd/SF/SF.s1337.mxsz250000.mnsz32000.maxtok1400000.MU1200k.ufreq1.layer_norm.ngpu64/checkpoint_last.pt",
    },
    "ted.ls.fsh.swbd": {
        "200K": "/checkpoint/anuroops/fairseq/robustw2v/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU200k.ufreq2.layer_norm.ngpu32/checkpoint_best.pt",
        "400K": "/checkpoint/anuroops/fairseq/robustw2v/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU400k.ufreq2.layer_norm.ngpu32/checkpoint_best.pt",
        "800K": "/checkpoint/anuroops/fairseq/robustw2v/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU800k.ufreq2.layer_norm.ngpu32/checkpoint_best.pt",
        "1200K": "/checkpoint/anuroops/fairseq/robustw2v/w2v_steps_ted.ls.fsh.swbd/w2v_steps_ted.ls.fsh.swbd.s1337.mxsz250000.mnsz32000.maxtok1400000.MU1200k.ufreq2.layer_norm.ngpu32/checkpoint_best.pt",
    },
}
