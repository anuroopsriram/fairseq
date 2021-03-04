#!/usr/bin/env python

from pathlib import Path
import sys

# root = sys.path[0]
# dir = 'fairseq-py-dev'
# sys.path.append(root[:root.index(dir) + len(dir)])

import config
import sweep as sweep
from sweep import hyperparam as hp


def get_grid(args):
    if args.last:
        ckpt = f'{args.checkpoints_dir}/checkpoint_last.pt'
    else:
        ckpt = f'{args.checkpoints_dir}/checkpoint_best.pt'
    assert Path(ckpt).exists()

    return [
        hp('--finetune-from-model', ckpt),

        # common
        hp('--fp16', True, binary_flag=True),
        hp('--log-format', 'json'),
        hp('--log-interval', 10 if args.local else 100),
        hp('--seed', [1337], save_dir_key=lambda val: f's{val}'),

        # checkpoint
        # hp('--save-interval-updates', 1000 if args.local else 100000),
        hp('--keep-interval-updates', -1),
        hp('--no-epoch-checkpoints'),
        hp('--no-save', True if args.local else False, binary_flag=True),

        # distributed_training
        hp('--ddp-backend', 'no_c10d'),
        hp('--find-unused-parameters'),

        # task
        hp('--task', 'audio_pretraining'),
        hp('--max-sample-size', 250000),  # , save_dir_key=lambda val: f'mxsz{val}'),
        hp('--min-sample-size', 32000),  # , save_dir_key=lambda val: f'mnsz{val}'),

        # dataset
        hp('--num-workers', 6),
        hp('--max-tokens', 700000 if args.local else 1400000),  # , save_dir_key=lambda val: f'maxtok{val}'),
        hp('--skip-invalid-size-inputs-valid-test', True, binary_flag=True),
        hp('--validate-interval-updates', 100 if args.local else 10000),
        hp('--validate-interval', 5),
        hp('--train-subset', config.DATASETS[args.dataset]["train"]),
        hp('--valid-subset', config.DATASETS[args.dataset]["val"]),

        # criterion
        hp('--criterion', 'wav2vec'),
        hp('--infonce', True, binary_flag=True),
        hp('--log-keys', '["prob_perplexity", "code_perplexity", "temp"]'),
        hp('--loss-weights', '[0.1, 10]'),

        # optimization
        hp('--max-update', [
            25000,
            # 50000,
            # 100000,
            # 200000,
            # 400000,
        ], save_dir_key=lambda val: f'MU{val//1000}k'),
        hp('--update-freq', [args.update_freq], save_dir_key=lambda val: f'ufreq{val}'),
        hp('--lr', [
            # 0.0005,
            # 0.0001,
            # 0.00005,
            0.00001,
            0.000001,
        ], save_dir_key=lambda val: f'lr{val}'),
        hp('--clip-norm', [25.0]),

        # optimizer
        hp('--optimizer', 'adam'),
        hp('--adam-betas', '(0.9,0.98)'),
        hp('--adam-eps', 1e-06),
        hp('--weight-decay', [0.01]),

        # lr_scheduler
        hp('--lr-scheduler', 'polynomial_decay'),

        # model
        hp('--arch', 'wav2vec2'),
        hp('--quantize-targets', True, binary_flag=True),
        hp('--final-dim', 256),
        hp("--encoder-layerdrop", [0.05,]),
        hp('--dropout', [0.1,]),
        hp('--dropout-input', [0.1]),
        hp('--dropout-features', [0.1]),
        hp('--attention-dropout', [0.1]),
        hp('--feature-grad-mult', [0.1]),
        hp('--extractor-mode', ['layer_norm'])  # , save_dir_key=lambda v: v),
    ]


def postprocess_hyperparams(args, config):
    max_update = config['--max-update'].current_value

    key = '--total-num-update'
    config[key] = hp(key, [])
    config[key].current_value = max_update

    key = '--warmup-updates'
    config[key] = hp(key, [])
    config[key].current_value = int(max_update * 0.08)

    extractor_mode = config.get("--extractor-mode", None)
    if extractor_mode is not None:
        key = '--normalize'
        config[key] = hp(key, [], binary_flag=True)
        config[key].current_value = (
            True if extractor_mode.current_value == "layer_norm" else False
        )


if __name__ == '__main__':
    sweep.main(get_grid, postprocess_hyperparams)
