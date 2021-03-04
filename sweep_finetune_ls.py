#!/usr/bin/env python

import os
import sys
from functools import reduce

# root = sys.path[0]
# dir = "fairseq-py-dev"
# sys.path.append(root[: root.index(dir) + len(dir)])

import config
import sweep
from sweep import hyperparam as hp


def get_lr_str(val):
    uniq_lr = reduce(
        lambda x, y: x + y if x[-1] != y[-1] else x, map(lambda x: [x], val.split(","))
    )
    return ",".join(uniq_lr)

def get_filter_str(val):
    f = eval(val)
    s = f"{f[0][1]}_{len(f)}"
    return s

def signature(val):
    import regex
    m = regex.match('.*_q\d?/([^\.]+)\..*', val)
    assert m
    return f'{m.group(1)}'

# lm_bin = "/checkpoint/abdo/old_checkpoint02/datasets/librispeech/4-gram.bin"
# lexicon = "/checkpoint/abdo/old_checkpoint02/datasets/librispeech/10h/raw/lexicon_ltr.lst"

def get_grid(args):
    if args.last:
        ckpt = f'{args.checkpoints_dir}/checkpoint_last.pt'
    else:
        ckpt = f'{args.checkpoints_dir}/checkpoint_best.pt'

    return [
        hp('--ddp-backend', 'no_c10d'),
        hp("--fp16", [True], binary_flag=True),
        hp('--num-workers', 8),
        
        # hp("--train-subset", "train_10h"),
        # hp("--valid-subset", "dev_other"),
        hp('--train-subset', config.LAB_DATASETS[args.dataset]["train"]),
        hp('--valid-subset', config.LAB_DATASETS[args.dataset]["val"]),
        hp("--task", "audio_pretraining"),
        hp("--criterion", "ctc"),
        hp("--arch", "wav2vec_ctc"),
        hp("--w2v-path", ckpt),
        hp("--no-pretrained-weights", False, binary_flag=True),

        # hp("--wer-args", f"('{lm_bin}','{lexicon}',2,-1)"),
        hp('--normalize', [
            # False,
            True,
        ], binary_flag=True, save_dir_key=lambda _: "norm"),
        hp("--sentence-avg", True, binary_flag=True),
        hp("--labels", ["ltr",]),

        hp("--optimizer", "adam"),
        hp("--adam-betas", "(0.9, 0.98)"),
        hp("--adam-eps", 1e-8),
        hp("--lr", [
            # 1e-5,
            2e-5,
            # 5e-5,
            # 8e-5,
        ], save_dir_key=lambda val: f"lr{val}"),
        hp("--lr-scheduler", ["tri_stage"]),
        hp("--max-update", 
            # config.LAB_DATASETS[args.dataset]["steps"], 
            [16000],
            save_dir_key=lambda val: f"u{val}"),
        hp("--warmup-steps", 
            # int(config.LAB_DATASETS[args.dataset]["steps"] * .2),
            [1600],
            save_dir_key=lambda val: f"wstep{val}"),
        hp("--hold-steps", 
            # int(config.LAB_DATASETS[args.dataset]["steps"] * .3), 
            [0],
            save_dir_key=lambda val: f"hstep{val}"),
        hp("--decay-steps", 
            # int(config.LAB_DATASETS[args.dataset]["steps"] * .5),
            [14400],
            save_dir_key=lambda val: f"dstep{val}"),
        hp("--final-lr-scale", [0.05], save_dir_key=lambda val: f"lrfs{val}"),
        hp('--update-freq', [1], save_dir_key=lambda val: f"ufreq{val}"),
        hp("--max-tokens", [
            # 1280000,
            3200000,
        ], save_dir_key=lambda val: f"maxtok{val}"),
        hp("--freeze-finetune-updates", 0, save_dir_key=lambda val: f"ffu{val}"),
        hp('--validate-after-updates', 0),
        hp("--feature-grad-mult", 0.0, save_dir_key=lambda val: f"fgm{val}"),

        hp("--apply-mask", True, binary_flag=True,),
        hp("--mask-selection", ["static"],),
        hp('--mask-length', [
            3,
            5,
            # 8,
            10,
            # 12,
            # 15,
        ], save_dir_key=lambda val: f"ml{val}"),
        hp('--mask-other', 0),
        hp('--mask-prob', [
            # 0.25,
            0.5,
            # 0.75,
        ], save_dir_key=lambda val: f'mp{val}'),
        hp("--mask-channel-selection", ["static"], save_dir_key=lambda v: f"mc{v}"),
        hp('--mask-channel-length', [64,], save_dir_key=lambda v: f"mcl{v}"),
        hp('--mask-channel-other', [0,], save_dir_key=lambda v: f"mco{v}"),
        hp('--mask-channel-prob', [0.5], save_dir_key=lambda val: f'mcp{val}'),
        hp("--zero-infinity", [True,], binary_flag=True,),

        hp("--layerdrop", [0.1], save_dir_key=lambda val: f"drpl{val}"),  # override encoder-layerdrop
        hp("--final-dropout",[0.0,]),  # apply to wav2vec2 features
        hp("--dropout",[0.0]),
        hp("--activation-dropout",[0.1]),
        hp("--attention-dropout",[0.0]),

        hp('--find-unused-parameters'),  # needed for distributed training
        hp("--no-epoch-checkpoints"),
        hp("--best-checkpoint-metric", "wer"),
        hp('--validate-interval-updates', 500),
        hp('--validate-interval', 10000),
        hp('--save-interval-updates', 500),
        hp('--save-interval', 10000),
        hp('--keep-interval-updates', 1),
        hp('--no-save', True if args.local else False, binary_flag=True),
        hp("--log-format", "json"),
        hp('--log-interval', 1 if args.local else 100),
        hp('--seed', [1337], save_dir_key=lambda val: f'sd{val}'),
        ############
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    if '--total-num-update' in config:
        config['--total-num-update'].current_value = config['--max-update'].current_value
    if '--freeze-finetune-updates' in config:
        config['--validate-after-updates'].current_value = config['--freeze-finetune-updates'].current_value


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
