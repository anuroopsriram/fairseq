from copy import deepcopy
from pathlib import Path

import numpy as np

import submit

wer_args = (
    '/datasets01/librispeech/021419/lm/4-gram.arpa',
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
    'mask-prob': 0.5,                       # 0.65
    'layerdrop': 0.1,                       # 0.05
    'mask-channel-selection': 'static',
    'mask-channel-other': 0,
    'mask-channel-length': 64,
    'mask-channel-prob': 0.5,
    'zero-infinity': True,
    'feature-grad-mult': 0.0,
    # 'freeze-finetune-updates': 10000,
    'freeze-finetune-updates': 0,           # 10000
    'validate-after-updates': 10000,
    'optimizer': 'adam',
    'adam-betas': (0.9, 0.98),
    'adam-eps': 1e-08,
    'lr': 2e-05,                            # 5e-5
    'lr-scheduler': 'tri_stage',
    'warmup-steps': 8000,                   # 0.1
    'hold-steps': 32000,                    # 0.4
    'decay-steps': 40000,                   # 0.5
    'final-lr-scale': 0.05,
    'final-dropout': 0.0,
    'dropout': 0.0,
    'activation-dropout': 0.1,
    'criterion': 'ctc',
    'attention-dropout': 0.0,
    # 'max-tokens': 1280000,
    # 'max-tokens': 4_000_000,
    'max-tokens': 2_000_000,                # 3200000
    'seed': 2337,                           # 1
    # 'log-format': 'json',
    'log-interval': 500,                    # 200
    'ddp-backend': 'no_c10d',               # legacy_ddp
    'validate-interval-updates': 500,       # 0
    'validate-interval': 10000,             # 50
    'save-interval-updates': 500,           # 10000
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
        "w2v.base.mlp.augment.8x400.ft": [
            # "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.15.unlab",
            # "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.15.unlab",
            # "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025augSrc1.0.augTgt1.0.augsadditive,speed.snr-min6_snr-max15_speed-std0.15.unlab",

            # "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            #### "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            # "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augsadditive,speed.snr-min6_snr-max15_speed-std0.15.unlab",

            # "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.0.ld0.0augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.15.unlab",
            # "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025augSrc1.0.augTgt1.0.augsadditive,speed.snr-min6_snr-max15_speed-std0.15.unlab",
            # "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.15.unlab",
            # "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.15.unlab",

            # "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            # "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.15.unlab",
            # "logs/w2v.base.mlp.augment.8x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
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
        ],
        "w2v.base.conf.8x400.ft": [
            # "logs/w2v.base.conf.8x400/lr0.0005.transconf.ksz3.cmlpFalse.tmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025.normFalseaugSrc1.0.augTgt1.0.augs{augmentations}.snr-min8_snr-max15_speed-std0.1.unlab",
            # "logs/w2v.base.conf.8x400/lr0.0005.transconf_rp.ksz3.cmlpFalse.tmlpTrue.bnTrue.actrelu.scale4.do0.05.ld0.025.normFalseaugSrc1.0.augTgt1.0.augs{augmentations}.snr-min8_snr-max15_speed-std0.1.unlab",
        ],
        "w2v.base.lc.2x100.ft": [
            # "logs/w2v.base.lc.2x100/lr0.0005.ctype_dc_last2.unlab",
            # "logs/w2v.base.lc.2x100/lr0.0005.ctype_lc_last2.8hds.unlab",
            # "logs/w2v.base.lc.2x100/lr0.0005.ctype_dc_last4.unlab",
            # "logs/w2v.base.lc.2x100/lr0.0005.ctype_dc_last6.unlab",
            # "logs/w2v.base.lc.2x100/lr0.0005.ctype_lc_last2.unlab",
            # "logs/w2v.base.lc.2x100/lr0.0005.ctype_lc_last4.unlab",
            # "logs/w2v.base.lc.2x100/lr0.0005.ctype_lc_last6.unlab",

            # "logs/w2v.base.lc.2x100/lr0.0005.ctype_conv.8hds.unlab",   # Baseline with new FE shapes
            # "logs/w2v.base.lc.2x100/lr0.0005.ctype_lc_last2.8hds.lconvparams4hds.unlab",
            # "logs/w2v.base.lc.2x100/lr0.0005.ctype_lc_last2.8hds.lconvparams8hds.unlab",
            # "logs/w2v.base.lc.2x100/lr0.0005.ctype_lc_last2.8hds.lconvparamsnoavgpool.unlab",
            # "logs/w2v.base.lc.2x100/lr0.0005.ctype_lc_last2.8hds.lconvparamsnodo.unlab",
            # "logs/w2v.base.lc.2x100/lr0.0005.ctype_lc_last2.8hds.lconvparamsnoglu.unlab",
            # "logs/w2v.base.lc.2x100/lr0.0005.ctype_lc_last2.8hds.lconvparamsnonormbef.unlab",
            # "logs/w2v.base.lc.2x100/lr0.0005.ctype_lc_last2.8hds.lconvparamsnowtsmax.unlab",

            # "logs/w2v.base.lc.2x100/lr0.0005.ctype_lc_last4.lconvparamsnoavgpool.unlab",
            # "logs/w2v.base.lc.2x100/lr0.0005.ctype_lc_last4.lconvparamsnowtsmax.unlab",
        ],
        "w2v.base.lc.8x400.ft": [
            # "logs/w2v.base.lc.8x400/lr0.0005.ctype_lc_last2.lconvparamsnoavgpool.unlab",
        ],
        "w2v.base.augment.2x100.100h.ft": [
            # "logs/w2v.base.augment.2x100.100h/add_8_15.lr0.0005.do0.0.ld0.0.unlab.100",
            # "logs/w2v.base.augment.2x100.100h/add_8_15.lr0.0005.do0.1.ld0.05.unlab.100",
            # "logs/w2v.base.augment.2x100.100h/baseline.lr0.0005.do0.0.ld0.0.unlab.100",
            # "logs/w2v.base.augment.2x100.100h/baseline.lr0.0005.do0.1.ld0.05.unlab.100",
            # "logs/w2v.base.augment.2x100.100h/spd0.1.lr0.0005.do0.0.ld0.0.unlab.100",
            "logs/w2v.base.augment.2x100.100h/spd0.1.lr0.0005.do0.1.ld0.05.unlab.100",
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
    lrs = [5e-05]
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
    # mask_lens = [3, 6, 10]
    # mask_lens = [3, 6, 10]
    mask_lens = [10]
    mask_probs = [0.5]
    dos = [0.1]

    for name, checkpoints_list in checkpoints.items():
        for checkpoint in checkpoints_list:
            args = deepcopy(base_args)
            args.nodes = 1
            # args.nodes = 3
            args.name = args.name or name
            checkpoint = Path(checkpoint)

            param_sweeps = [
                (
                    f"ckpt{checkpoint.name}.lr{lr}.mlen{mlen}.mprob{mprob}.do{do}.ngram",
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

                        'wer-args': wer_args,
                        # "normalize": True,
                    }
                )
                for do in dos
                for mlen in mask_lens
                for mprob in mask_probs
                for lr in lrs
            ]
            submit.run_sweeps(args, base_params, param_sweeps, dataset='lab.10h', skip_if_cp_exists=False)
            # submit.run_sweeps(args, base_params, param_sweeps, dataset='lab.10h', skip_if_cp_exists=True)


@submit.register_sweep
def w2v_base_mlp_augment_ablation(base_args):
    checkpoints = {
        "w2v.base.mlp.augment.4x400.ft": [
            # # Baseline
            # "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpFalse.tgtmlpFalse.bnFalse.actrelu.scale1.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab",
            # # Aug, No MLP
            # "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpFalse.tgtmlpFalse.bnFalse.actrelu.scale1.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            # # Target MLP
            # "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab",
            # # Context MLP
            # "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpTrue.tgtmlpFalse.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab",
            # # Context MLP + Target MLP
            # "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab",
            # Target MLP + Aug
            "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            # # Context MLP + Aug
            # "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpTrue.tgtmlpFalse.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            # # Context MLP + Target MLP + Aug
            # "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
        ],
        # "w2v.base.mlp.augment.search.4x400.ft": [
        #     "logs/w2v.base.mlp.augment.search.4x400/lr0.0005.tgtmlp.augSrc1.0.augTgt1.0.augsadditive,speed,pitch.snr-min8_snr-max15_speed-std0.1_pitch-shift-std10.unlab",
        #     "logs/w2v.base.mlp.augment.search.4x400/lr0.0005.tgtmlp.augSrc1.0.augTgt1.0.augsadditive,speed,reverb.snr-min8_snr-max15_speed-std0.1_reverb-strength30.unlab",
        #     "logs/w2v.base.mlp.augment.search.4x400/lr0.0005.tgtmlp.augSrc1.0.augTgt1.0.augsadditive,speed.snr-min5_snr-max15_speed-std0.1.unlab",
        #     "logs/w2v.base.mlp.augment.search.4x400/lr0.0005.tgtmlp.augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
        #     "logs/w2v.base.mlp.augment.search.4x400/lr0.0005.tgtmlp.augSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.15.unlab",
        # ],
    }
    mask_lens = [3, 6, 10]
    mask_probs = [0.5]
    dos = [0.1]
    max_update = 25_000
    lrs = [
        1e-05,
        2e-05,
    ]
    for name, checkpoints_list in checkpoints.items():
        for checkpoint in checkpoints_list:
            checkpoint = Path(checkpoint)
            args = deepcopy(base_args)
            args.nodes = 2
            args.name = (args.name or name) + "/" + checkpoint.name

            param_sweeps = [
                (
                    f"lr{lr}.mlen{mlen}.mprob{mprob}.do{do}.ngram",
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

                        'wer-args': wer_args,
                    }
                )
                for do in dos
                for mlen in mask_lens
                for mprob in mask_probs
                for lr in lrs
            ]
            submit.run_sweeps(args, base_params, param_sweeps, dataset='lab.10h', skip_if_cp_exists=False)


@submit.register_sweep
def w2v_base_mlp_augment_ablation2(base_args):
    checkpoints = {
        "w2v.base.mlp.augment.4x400.ft": [
            # # Baseline
            # "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpFalse.tgtmlpFalse.bnFalse.actrelu.scale1.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab",
            # # Aug, No MLP
            # "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpFalse.tgtmlpFalse.bnFalse.actrelu.scale1.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            # # Target MLP
            # "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab",
            # # Context MLP
            # "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpTrue.tgtmlpFalse.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab",
            # # Context MLP + Target MLP
            # "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab",
            # Target MLP + Aug
            "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            # # Context MLP + Aug
            # "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpTrue.tgtmlpFalse.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            # # Context MLP + Target MLP + Aug
            # "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
        ],
    }
    mask_lens = [
        # 3,
        # 6,
        10,
        12,
        15,
    ]
    mask_probs = [
        # 0.45,
        0.65,
        0.75,
    ]
    max_update = 20_000
    lrs = [
        # 2e-05,
        5e-05,
        # 8e-05,
    ]
    lds = [
        0.,
        0.05,
        0.1
    ]
    for name, checkpoints_list in checkpoints.items():
        for checkpoint in checkpoints_list:
            checkpoint = Path(checkpoint)
            args = deepcopy(base_args)
            # args.nodes = 2
            args.nodes = 1
            args.name = (args.name or name) + "/" + checkpoint.name

            param_sweeps = [
                (
                    f"lr{lr}.mlen{mlen}.mprob{mprob}.ld{ld}.ngram.1nd",
                    {
                        "lr": lr,
                        'mask-length': mlen,
                        'mask-prob': mprob,

                        "max-update": max_update,
                        "freeze-finetune-updates": 10000,
                        "warmup-steps": int(max_update * 0.1),
                        "hold-steps": int(max_update * 0.4),
                        "decay-steps": int(max_update * 0.5),
                        "w2v-path": checkpoint / "checkpoint_best.pt",
                        "augment-audio": False,
                        'layerdrop': ld,
                        'wer-args': wer_args,
                        'max-tokens': 3_200_000,
                        'log-interval': 200,
                        'validate-interval-updates': 0,
                        'validate-interval': 50,
                        'save-interval-updates': 10000,
                    }
                )
                for ld in lds
                for mlen in mask_lens
                for mprob in mask_probs
                for lr in lrs
            ]
            submit.run_sweeps(args, base_params, param_sweeps, dataset='lab.10h', skip_if_cp_exists=False)


@submit.register_sweep
def w2v_base_glu(base_args):
    checkpoints = {
        "w2v.base.glu.4x400.ft100": [
            "logs/w2v.base.glu.4x400/geglu.lr0.0002.unlab",
            # "logs/w2v.base.glu.4x400/geglu.lr0.0005.unlab",
            "logs/w2v.base.glu.4x400/geglu.lr0.001.unlab",
            "logs/w2v.base.glu.4x400/geglu_bias.lr0.0002.unlab",
            "logs/w2v.base.glu.4x400/geglu_bias.lr0.0005.unlab",
            "logs/w2v.base.glu.4x400/swish.lr0.0002.unlab",
            "logs/w2v.base.glu.4x400/swish.lr0.0005.unlab",
            "logs/w2v.base.glu.4x400/swish.lr0.001.unlab",
            "logs/w2v.base.glu.4x400/swish_bias.lr0.0002.unlab",
            "logs/w2v.base.glu.4x400/swish_bias.lr0.0005.unlab",
        ],
    }
    mask_lens = [
        # 6,
        10,
        # 13,
    ]
    mask_probs = [
        0.45,
        # 0.65,
    ]
    max_update = 80_000
    lrs = [
        2e-05,
        5e-05,
    ]
    for name, checkpoints_list in checkpoints.items():
        for checkpoint in checkpoints_list:
            checkpoint = Path(checkpoint)
            args = deepcopy(base_args)
            # args.nodes = 2
            args.nodes = 1
            args.name = (args.name or name) + "/" + checkpoint.name

            param_sweeps = [
                (
                    f"lr{lr}.mlen{mlen}.mprob{mprob}.ngram.1nd",
                    {
                        "lr": lr,
                        'mask-length': mlen,
                        'mask-prob': mprob,

                        "max-update": max_update,
                        "freeze-finetune-updates": 10000,
                        "warmup-steps": int(max_update * 0.1),
                        "hold-steps": int(max_update * 0.4),
                        "decay-steps": int(max_update * 0.5),
                        "w2v-path": checkpoint / "checkpoint_best.pt",
                        "augment-audio": False,
                        'layerdrop': 0.05,
                        'wer-args': wer_args,
                        'max-tokens': 3_200_000,
                        'log-interval': 200,
                        'validate-interval-updates': 0,
                        'validate-interval': 50,
                        'save-interval-updates': 10000,
                    }
                )
                for mlen in mask_lens
                for mprob in mask_probs
                for lr in lrs
            ]
            submit.run_sweeps(args, base_params, param_sweeps, dataset='lab.100h', skip_if_cp_exists=False)


@submit.register_sweep
def w2v_base_mlp_augment_ablation_100h(base_args):
    checkpoints = {
        "w2v.base.mlp.augment.4x400.ft100": [
            # # Baseline
            # "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpFalse.tgtmlpFalse.bnFalse.actrelu.scale1.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab",
            # # Aug, No MLP
            # "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpFalse.tgtmlpFalse.bnFalse.actrelu.scale1.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            # # Target MLP
            # "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab",
            # # Context MLP
            # "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpTrue.tgtmlpFalse.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab",
            # # Context MLP + Target MLP
            # "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab",
            # Target MLP + Aug
            "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            # # Context MLP + Aug
            # "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpTrue.tgtmlpFalse.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
            # # Context MLP + Target MLP + Aug
            # "logs/w2v.base.mlp.augment.4x400/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab",
        ],
    }
    mask_lens = [
        # 3,
        # 6,
        10,
        # 13,
    ]
    mask_probs = [
        # 0.35,
        # 0.40,
        0.45,
        # 0.50,
        # 0.55,
        # 0.65,
        # 0.75,
    ]
    dos = [0.1]
    max_update = 80_000
    lrs = [
        2e-05,
        # 3e-05,
        5e-05,
    ]
    for name, checkpoints_list in checkpoints.items():
        for checkpoint in checkpoints_list:
            checkpoint = Path(checkpoint)
            args = deepcopy(base_args)
            args.nodes = 1
            args.name = (args.name or name) + "/" + checkpoint.name

            param_sweeps = [
                (
                    f"lr{lr}.mlen{mlen}.mprob{mprob}.do{do}.ngram",
                    {
                        "lr": lr,
                        'mask-length': mlen,
                        'mask-prob': mprob,

                        "max-update": max_update,
                        "warmup-steps": int(max_update * 0.1),
                        "hold-steps": int(max_update * 0.5),
                        "decay-steps": int(max_update * 0.3),
                        "w2v-path": checkpoint / "checkpoint_best.pt",

                        "augment-audio": False,
                        'layerdrop': do,
                        # 'final-dropout': do,
                        # 'dropout': do,
                        'activation-dropout': do,
                        # 'attention-dropout': do,
                        'freeze-finetune-updates': 0,

                        'wer-args': wer_args,

                        'max-tokens': 3_200_000,
                        'log-interval': 200,
                    }
                )
                for do in dos
                for mlen in mask_lens
                for mprob in mask_probs
                for lr in lrs
            ]
            submit.run_sweeps(args, base_params, param_sweeps, dataset='lab.100h', skip_if_cp_exists=False)


if __name__ == '__main__':
    parser = submit.create_parser()
    base_args = parser.parse_args()
    sweep_func = submit.get_sweep_func(base_args.sweep)
    sweep_func(base_args)
