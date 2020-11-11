from pathlib import Path

import submit

wer_args = (
    '/datasets01_101/librispeech/021419/lm/4-gram.arpa',
    '/checkpoint/anuroops/data/libris/lab.960h/lexicon_ltr.lst',
    2, -1,
)


base_params = {
    'distributed-world-size': 24,
    'distributed-port': 13434,
    'save-dir': '/checkpoint/anuroops/fairseq/wav2vec/w2v.base.ft/',
    'fp16': True,
    'post-process': 'letter',
    'valid-subset': 'dev_other',
    'no-epoch-checkpoints': True,
    'best-checkpoint-metric': 'wer',
    'num-workers': 4,
    'max-update': 80000,
    'sentence-avg': True,
    'task': 'audio_pretraining',
    'arch': 'wav2vec_ctc',
    'labels': 'ltr',
    'apply-mask': True,
    'mask-selection': 'static',
    'mask-other': 0,
    'mask-length': 10,
    'mask-prob': 0.5,
    'layerdrop': 0.1,
    'mask-channel-selection': 'static',
    'mask-channel-other': 0,
    'mask-channel-length': 64,
    'mask-channel-prob': 0.5,
    'zero-infinity': True,
    'feature-grad-mult': 0.0,
    # 'freeze-finetune-updates': 10000,
    'freeze-finetune-updates': 0,
    'validate-after-updates': 10000,
    'optimizer': 'adam',
    'adam-betas': (0.9, 0.98),
    'adam-eps': 1e-08,
    'lr': 2e-05,
    'lr-scheduler': 'tri_stage',
    'warmup-steps': 8000,
    'hold-steps': 32000,
    'decay-steps': 40000,
    'final-lr-scale': 0.05,
    'final-dropout': 0.0,
    'dropout': 0.0,
    'activation-dropout': 0.1,
    'criterion': 'ctc',
    'attention-dropout': 0.0,
    'max-tokens': 1280000,
    'seed': 2337,
    'log-format': 'json',
    'log-interval': 500,
    'ddp-backend': 'no_c10d',

    'augment-audio': False,
    "normalize": True,
}


def w2v_base_2x100k(args, params):
    args.name = args.name or 'w2v.base.2x100k.augment.ft'
    args.nodes = 2
    return args, params


#### Sweeps


@submit.register_sweep
def sweep_w2v_base_2x100k(base_args):
    checkpoints = {
        # # No Augmentation
        # "lr0.0005.noaug.unlab": Path("logs/w2v.base.100k.augment/lr0.0005.noaug.unlab"),
        # # Augmentations LayerDrop = 0.05
        # "lr0.0005.sourceaug1.targetaug0.0.augsadditive": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug1.targetaug0.0.augsadditive.unlab"),
        # "lr0.0005.sourceaug0.0.targetaug1.augsadditive": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug0.0.targetaug1.augsadditive.unlab"),
        # "lr0.0005.sourceaug1.targetaug1.augsadditive": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug1.targetaug1.augsadditive.unlab"),
        # "lr0.0005.sourceaug1.targetaug0.0.augsspeed": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug1.targetaug0.0.augsspeed.unlab"),
        # "lr0.0005.sourceaug0.0.targetaug1.augsspeed": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug0.0.targetaug1.augsspeed.unlab"),
        # "lr0.0005.sourceaug1.targetaug1.augsspeed": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug1.targetaug1.augsspeed.unlab"),
        # "lr0.0005.sourceaug1.targetaug0.0.augspitch": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug1.targetaug0.0.augspitch.unlab"),
        # "lr0.0005.sourceaug0.0.targetaug1.augspitch": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug0.0.targetaug1.augspitch.unlab"),
        # "lr0.0005.sourceaug1.targetaug1.augspitch": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug1.targetaug1.augspitch.unlab"),
        # "lr0.0005.sourceaug1.targetaug0.0.augsreverb": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug1.targetaug0.0.augsreverb.unlab"),
        # "lr0.0005.sourceaug0.0.targetaug1.augsreverb": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug0.0.targetaug1.augsreverb.unlab"),
        # "lr0.0005.sourceaug1.targetaug1.augsreverb": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug1.targetaug1.augsreverb.unlab"),
        # "lr0.0005.sourceaug1.targetaug0.0.augsadditive,pitch,speed,reverb": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug1.targetaug0.0.augsadditive,pitch,speed,reverb.unlab"),
        # "lr0.0005.sourceaug0.0.targetaug1.augsadditive,pitch,speed,reverb": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug0.0.targetaug1.augsadditive,pitch,speed,reverb.unlab"),
        # "lr0.0005.sourceaug1.targetaug1.augsadditive,pitch,speed,reverb": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug1.targetaug1.augsadditive,pitch,speed,reverb.unlab"),
        # # Augmentations LayerDrop = 0.
        # "lr0.0005.sourceaug1.targetaug0.0.augsadditive.ld0.0": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug1.targetaug0.0.augsadditive.ld0.0.unlab"),
        # "lr0.0005.sourceaug0.0.targetaug1.augsadditive.ld0.0": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug0.0.targetaug1.augsadditive.ld0.0.unlab"),
        # "lr0.0005.sourceaug1.targetaug1.augsadditive.ld0.0": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug1.targetaug1.augsadditive.ld0.0.unlab"),
        # "lr0.0005.sourceaug1.targetaug0.0.augsspeed.ld0.0": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug1.targetaug0.0.augsspeed.ld0.0.unlab"),
        # "lr0.0005.sourceaug0.0.targetaug1.augsspeed.ld0.0": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug0.0.targetaug1.augsspeed.ld0.0.unlab"),
        # "lr0.0005.sourceaug1.targetaug1.augsspeed.ld0.0": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug1.targetaug1.augsspeed.ld0.0.unlab"),
        # "lr0.0005.sourceaug1.targetaug0.0.augspitch.ld0.0": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug1.targetaug0.0.augspitch.ld0.0.unlab"),
        # "lr0.0005.sourceaug0.0.targetaug1.augspitch.ld0.0": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug0.0.targetaug1.augspitch.ld0.0.unlab"),
        # "lr0.0005.sourceaug1.targetaug1.augspitch.ld0.0": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug1.targetaug1.augspitch.ld0.0.unlab"),
        # "lr0.0005.sourceaug1.targetaug0.0.augsreverb.ld0.0": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug1.targetaug0.0.augsreverb.ld0.0.unlab"),
        # "lr0.0005.sourceaug0.0.targetaug1.augsreverb.ld0.0": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug0.0.targetaug1.augsreverb.ld0.0.unlab"),
        # "lr0.0005.sourceaug1.targetaug1.augsreverb.ld0.0": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug1.targetaug1.augsreverb.ld0.0.unlab"),
        # "lr0.0005.sourceaug1.targetaug0.0.augsadditive,pitch,speed,reverb.ld0.0": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug1.targetaug0.0.augsadditive,pitch,speed,reverb.ld0.0.unlab"),
        # "lr0.0005.sourceaug0.0.targetaug1.augsadditive,pitch,speed,reverb.ld0.0": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug0.0.targetaug1.augsadditive,pitch,speed,reverb.ld0.0.unlab"),
        # "lr0.0005.sourceaug1.targetaug1.augsadditive,pitch,speed,reverb.ld0.0": Path("logs/w2v.base.100k.augment/lr0.0005.sourceaug1.targetaug1.augsadditive,pitch,speed,reverb.ld0.0.unlab"),
        # No Quant
        "noquant.lr0.0005.sourceaug1.targetaug0.0.augsadditive": Path("logs/w2v.base.100k.augment/noquant.lr0.0005.sourceaug1.targetaug0.0.augsadditive.unlab"),
        "noquant.lr0.0005.sourceaug0.0.targetaug1.augsadditive": Path("logs/w2v.base.100k.augment/noquant.lr0.0005.sourceaug0.0.targetaug1.augsadditive.unlab"),
        "noquant.lr0.0005.sourceaug1.targetaug1.augsadditive": Path("logs/w2v.base.100k.augment/noquant.lr0.0005.sourceaug1.targetaug1.augsadditive.unlab"),
        "noquant.lr0.0005.sourceaug1.targetaug0.0.augsspeed": Path("logs/w2v.base.100k.augment/noquant.lr0.0005.sourceaug1.targetaug0.0.augsspeed.unlab"),
        "noquant.lr0.0005.sourceaug0.0.targetaug1.augsspeed": Path("logs/w2v.base.100k.augment/noquant.lr0.0005.sourceaug0.0.targetaug1.augsspeed.unlab"),
        "noquant.lr0.0005.sourceaug1.targetaug1.augsspeed": Path("logs/w2v.base.100k.augment/noquant.lr0.0005.sourceaug1.targetaug1.augsspeed.unlab"),
        "noquant.lr0.0005.sourceaug1.targetaug0.0.augspitch": Path("logs/w2v.base.100k.augment/noquant.lr0.0005.sourceaug1.targetaug0.0.augspitch.unlab"),
        "noquant.lr0.0005.sourceaug0.0.targetaug1.augspitch": Path("logs/w2v.base.100k.augment/noquant.lr0.0005.sourceaug0.0.targetaug1.augspitch.unlab"),
        "noquant.lr0.0005.sourceaug1.targetaug1.augspitch": Path("logs/w2v.base.100k.augment/noquant.lr0.0005.sourceaug1.targetaug1.augspitch.unlab"),
        "noquant.lr0.0005.sourceaug1.targetaug0.0.augsreverb": Path("logs/w2v.base.100k.augment/noquant.lr0.0005.sourceaug1.targetaug0.0.augsreverb.unlab"),
        "noquant.lr0.0005.sourceaug0.0.targetaug1.augsreverb": Path("logs/w2v.base.100k.augment/noquant.lr0.0005.sourceaug0.0.targetaug1.augsreverb.unlab"),
        "noquant.lr0.0005.sourceaug1.targetaug1.augsreverb": Path("logs/w2v.base.100k.augment/noquant.lr0.0005.sourceaug1.targetaug1.augsreverb.unlab"),
        "noquant.lr0.0005.sourceaug1.targetaug0.0.augsadditive,pitch,speed,reverb": Path("logs/w2v.base.100k.augment/noquant.lr0.0005.sourceaug1.targetaug0.0.augsadditive,pitch,speed,reverb.unlab"),
        "noquant.lr0.0005.sourceaug0.0.targetaug1.augsadditive,pitch,speed,reverb": Path("logs/w2v.base.100k.augment/noquant.lr0.0005.sourceaug0.0.targetaug1.augsadditive,pitch,speed,reverb.unlab"),
        "noquant.lr0.0005.sourceaug1.targetaug1.augsadditive,pitch,speed,reverb": Path("logs/w2v.base.100k.augment/noquant.lr0.0005.sourceaug1.targetaug1.augsadditive,pitch,speed,reverb.unlab"),
    }
    for checkpoint in checkpoints.values():
        assert checkpoint.exists(), checkpoint

    lrs = [4e-5]
    param_sweeps = [
        (
            f'{checkpoint.name}/{name}.lr{lr}',
            {
                'w2v-path': checkpoint / "checkpoint_best.pt",
                'lr': lr,
            },
        )
        for lr in lrs
        for name, checkpoint in checkpoints.items()
    ]
    submit.run_sweeps(w2v_base_2x100k, base_args, base_params, param_sweeps, dataset='lab.10h')


if __name__ == '__main__':
    parser = submit.create_parser()
    base_args = parser.parse_args()
    sweep_func = submit.get_sweep_func(base_args.sweep)
    sweep_func(base_args)
