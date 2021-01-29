
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
    }
}

LM = {
    "ls": "...",
}
