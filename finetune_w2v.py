from copy import deepcopy
from pathlib import Path

import numpy as np

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
    'max-update': 320000,
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
    # 'log-format': 'json',
    'log-interval': 500,
    'ddp-backend': 'no_c10d',
    'validate-interval-updates': 500,
    'validate-interval': 10000,
    # 'save-interval-updates': 500,
    'no-epoch-checkpoints': True,
}


@submit.register_sweep
def w2v_base_mlp(base_args):
    checkpoints = {
        "w2v.base.mlp.8x400.tmp": [
            # "logs/w2v.base.8x400.mlp/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.unlab",
        ],
        "w2v.base.mlp.4x400.tmp": [
            # "logs/w2v.base.mlp.4x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.unlab",
            # "logs/w2v.base.4x400.mlp/lr0.0005.contextmlpTrue.tgtmlpTrue.bnFalse.actrelu.unlab",
        ],
        "w2v.base.mlp.2x100.ft": [
            # "logs/w2v.base.mlp.2x100/lr0.0005.contextmlpFalse.tgtmlpFalse.bnFalse.actrelu.scale1.unlab",
            # "logs/w2v.base.mlp.2x100/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.unlab",
            # "logs/w2v.base.mlp.2x100/lr0.0005.contextmlpTrue.tgtmlpFalse.bnFalse.actrelu.scale4.unlab",
            # "logs/w2v.base.mlp.2x100/lr0.0005.contextmlpFalse.tgtmlpTrue.bnFalse.actrelu.scale4.unlab",
        ],
        "w2v.base.mlp.augment.2x100.ft": [
            # # contextmlpFalse.tgtmlpTrue
            # "logs/w2v.base.mlp.augment.2x100/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            # "logs/w2v.base.mlp.augment.2x100/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.15.unlab",
            # "logs/w2v.base.mlp.augment.2x100/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            # "logs/w2v.base.mlp.augment.2x100/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.15.unlab",
            # "logs/w2v.base.mlp.augment.2x100/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            # "logs/w2v.base.mlp.augment.2x100/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.15.unlab",
            # # contextmlpTrue.tgtmlpTrue
            # "logs/w2v.base.mlp.augment.2x100/lr0.0005.contextmlpTrue.tgtmlpFalse.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            # "logs/w2v.base.mlp.augment.2x100/lr0.0005.contextmlpTrue.tgtmlpFalse.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.15.unlab",
            # "logs/w2v.base.mlp.augment.2x100/lr0.0005.contextmlpTrue.tgtmlpFalse.bnTrue.actrelu.scale4.do0.05.ld0.025augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            # "logs/w2v.base.mlp.augment.2x100/lr0.0005.contextmlpTrue.tgtmlpFalse.bnTrue.actrelu.scale4.do0.05.ld0.025augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.15.unlab",
            # "logs/w2v.base.mlp.augment.2x100/lr0.0005.contextmlpTrue.tgtmlpFalse.bnTrue.actrelu.scale4.do0.1.ld0.05augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.15.unlab",
            # # contextmlpTrue.tgtmlpTrue
            # "logs/w2v.base.mlp.augment.2x100/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            # "logs/w2v.base.mlp.augment.2x100/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.15.unlab",
            # "logs/w2v.base.mlp.augment.2x100/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            # "logs/w2v.base.mlp.augment.2x100/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.15.unlab",
            # "logs/w2v.base.mlp.augment.2x100/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            # "logs/w2v.base.mlp.augment.2x100/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.15.unlab",
        ],
        "w2v.base.mlp.augment.8x400.ft.3x80k": [
            # "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.15.unlab",
            # "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.15.unlab",
            # "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025augSrc1.0.augTgt1.0.augsadditive,speed.snr-min6_snr-max15_speed-std0.15.unlab",

            "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augsadditive,speed.snr-min6_snr-max15_speed-std0.15.unlab",

            # "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.15.unlab",
            # "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025augSrc1.0.augTgt1.0.augsadditive,speed.snr-min6_snr-max15_speed-std0.15.unlab",
            # "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.15.unlab",
            # "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.15.unlab",
        ],
        "w2v.base.conf.2x100.ft": [
            # No MLP or Aug
            # "logs/w2v.base.conf.2x100/lr0.0005.transconf.unlab",
            # "logs/w2v.base.conf.2x100/lr0.0005.transconf_rp.unlab",
            # # No Aug
            # "logs/w2v.base.conf.2x100/lr0.0005.transconf.ksz3.cmlpFalse.tmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augs{augmentations}.speed-std0.0.unlab",
            # "logs/w2v.base.conf.2x100/lr0.0005.transconf.ksz3.cmlpTrue.tmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augs{augmentations}.speed-std0.0.unlab",
            # "logs/w2v.base.conf.2x100/lr0.0005.transconf_rp.ksz3.cmlpFalse.tmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augs{augmentations}.speed-std0.0.unlab",
            # "logs/w2v.base.conf.2x100/lr0.0005.transconf_rp.ksz3.cmlpTrue.tmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augs{augmentations}.speed-std0.0.unlab",
            # # MLP + Aug
            # "logs/w2v.base.conf.2x100/lr0.0005.transconf.ksz3.cmlpFalse.tmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augs{augmentations}.snr-min8_snr-max15_speed-std0.1.unlab",
            # "logs/w2v.base.conf.2x100/lr0.0005.transconf.ksz3.cmlpFalse.tmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augs{augmentations}.snr-min8_snr-max15_speed-std0.15.unlab",
            # "logs/w2v.base.conf.2x100/lr0.0005.transconf.ksz3.cmlpTrue.tmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augs{augmentations}.snr-min8_snr-max15_speed-std0.1.unlab",
            # "logs/w2v.base.conf.2x100/lr0.0005.transconf.ksz3.cmlpTrue.tmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augs{augmentations}.snr-min8_snr-max15_speed-std0.15.unlab",
            # "logs/w2v.base.conf.2x100/lr0.0005.transconf_rp.ksz3.cmlpFalse.tmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augs{augmentations}.snr-min8_snr-max15_speed-std0.1.unlab",
            # "logs/w2v.base.conf.2x100/lr0.0005.transconf_rp.ksz3.cmlpFalse.tmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augs{augmentations}.snr-min8_snr-max15_speed-std0.15.unlab",
            # "logs/w2v.base.conf.2x100/lr0.0005.transconf_rp.ksz3.cmlpTrue.tmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augs{augmentations}.snr-min8_snr-max15_speed-std0.1.unlab",
            # "logs/w2v.base.conf.2x100/lr0.0005.transconf_rp.ksz3.cmlpTrue.tmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augs{augmentations}.snr-min8_snr-max15_speed-std0.15.unlab",
        ]
    }

    # name = "w2v.base.mlp.2x100.ft"

    # checkpoints = {
    #     # "logs/w2v.base.mlp.2x100/lr0.0005.contextmlpFalse.tgtmlpFalse.bnFalse.actrelu.scale1.unlab",
    #     # "logs/w2v.base.mlp.2x100/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.unlab",
    #     "logs/w2v.base.mlp.2x100/lr0.0005.contextmlpTrue.tgtmlpFalse.bnFalse.actrelu.scale4.unlab",
    #     "logs/w2v.base.mlp.2x100/lr0.0005.contextmlpFalse.tgtmlpTrue.bnFalse.actrelu.scale4.unlab",
    # }
    max_update = 25_000
    # max_update = 80_000
    # lrs = [1e-05, 2e-05, 4e-05]
    lrs = [1e-05, 2e-05]
    # lrs = [1e-05, 4e-05]
    # mask_lens = [4, 10]
    # mask_probs = [0.5]

    # mask_lens = [4, 6, 10, 14]
    # mask_probs = [0.25, 0.5, 0.75]
    # dos = [0.1, 0.2]

    # mask_lens = [8, 10]
    # mask_probs = [0.15, 0.25, 0.4]
    # dos = [0.1, 0.2]

    # lrs = [1e-05]
    # mask_lens = [3, 4, 5, 7, 10]
    mask_lens = [3, 6, 10]
    mask_probs = [0.5]
    dos = [0.1]

    for name, checkpoints_list in checkpoints.items():
        for checkpoint in checkpoints_list:
            args = deepcopy(base_args)
            # args.nodes = 2
            args.nodes = 3
            args.name = args.name or name
            checkpoint = Path(checkpoint)

            param_sweeps = [
                (
                    f"ckpt{checkpoint.name}.lr{lr}.mlen{mlen}.mprob{mprob}.do{do}",
                    {
                        "lr": lr,
                        'mask-length': mlen,
                        'mask-prob': mprob,

                        "max-update": max_update,
                        "warmup-steps": int(max_update * 0.2),
                        "hold-steps": int(max_update * 0.5),
                        "decay-steps": int(max_update * 0.3),
                        "w2v-path": checkpoint / "checkpoint_best.pt",

                        "augment-audio": False,
                        'layerdrop': do,
                        'final-dropout': do,
                        'dropout': do,
                        'activation-dropout': do,
                        'attention-dropout': do,

                        # "normalize": True,
                    }
                )
                for do in dos
                for mlen in mask_lens
                for mprob in mask_probs
                for lr in lrs
            ]
            submit.run_sweeps(args, base_params, param_sweeps, dataset='lab.10h', skip_if_cp_exists=True)


if __name__ == '__main__':
    parser = submit.create_parser()
    base_args = parser.parse_args()
    sweep_func = submit.get_sweep_func(base_args.sweep)
    sweep_func(base_args)
