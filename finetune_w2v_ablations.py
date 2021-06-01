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
    'save-dir': '',
    'no-epoch-checkpoints': True,
    'fp16': True,
    'distributed-world-size': 24,
    'distributed-port': 13434,
    'post-process': 'letter',
    'valid-subset': 'dev_other',
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
    # 'validate-after-updates': 10000,
    'optimizer': 'adam',
    'adam-betas': (0.9, 0.98),
    'adam-eps': 1e-08,
    'lr': 2e-05,                            # 5e-5
    'lr-scheduler': 'tri_stage',
    # 'warmup-steps': 8000,                   # 0.1
    # 'hold-steps': 32000,                    # 0.4
    # 'decay-steps': 40000,                   # 0.5
    'final-lr-scale': 0.05,
    'final-dropout': 0.0,
    'dropout': 0.0,
    'activation-dropout': 0.1,
    'criterion': 'ctc',
    'attention-dropout': 0.0,
    # 'max-tokens': 3_200_000,                # 3200000
    'max-tokens': 4_800_000,                # 3200000
    'seed': 2337,                           # 1
    # 'log-format': 'json',
    'log-interval': 200,                    # 200
    'ddp-backend': 'no_c10d',               # legacy_ddp
    # 'validate-interval-updates': 500,       # 0
    # 'validate-interval': 10000,             # 50
    # 'save-interval-updates': 500,           # 10000
    'no-epoch-checkpoints': True,

    "distributed-no-spawn": True,
}


checkpoints = {
    # "ablation.baseline.ls960h.8x400.ft": [
    #     "logs/ablation.baseline.ls960h.8x400/ls960h.baselinelr0.0005.ls960h",
    # ],
    # # "ablation.aug.ls10h.3x100.ft": [
    # #     "logs/ablation.aug.ls10h.3x100/ls10h.add8.15lr0.0005.ls10h",
    # #     "logs/ablation.aug.ls10h.3x100/ls10h.noauglr0.0005.ls10h",
    # # ],
    # # "ablation.aug.50M.ls10h.3x100.ft": [
    # #     "logs/ablation.aug.50M.ls10h.3x100/ls10h.noauglr0.0005.ls10h",
    # #     "logs/ablation.aug.50M.ls10h.3x100/ls10h.add8.15lr0.0005.ls10h",
    # # ],
    "ablation.aug.ls50h.3x200.ft": [
        # "logs/ablation.aug.ls50h.3x150/ls50h.noauglr0.0005.ls50h",
        # "logs/ablation.aug.ls50h.3x150/ls50h.add8.15lr0.0005.ls50h",
        # "logs/ablation.aug.ls50h.3x150/ls50h.spd0.15lr0.0005.ls50h",
        # "logs/ablation.aug.ls50h.3x150/ls50h.spd0.10lr0.0005.ls50h",
        # "logs/ablation.aug.ls50h.3x150/ls50h.pitch20lr0.0005.ls50h",
        # "logs/ablation.aug.ls50h.3x150/ls50h.pitch50lr0.0005.ls50h",
        # "logs/ablation.aug.ls50h.3x150/ls50h.pitch100lr0.0005.ls50h",
        # "logs/ablation.aug.ls50h.3x150/ls50h.reverb10lr0.0005.ls50h",
        # "logs/ablation.aug.ls50h.3x150/ls50h.reverb25lr0.0005.ls50h",
        # "logs/ablation.aug.ls50h.3x150/ls50h.reverb60lr0.0005.ls50h",
        # "logs/ablation.aug.ls50h.3x150/ls50h.a8.15.p50.r25_p0.1lr0.0005.ls50h",
        # "logs/ablation.aug.ls50h.3x150/ls50h.a8.15.p50.r25_p0.3lr0.0005.ls50h",
        # "logs/ablation.aug.ls50h.3x150/ls50h.a8.15.p50.r25_p0.5lr0.0005.ls50h",
        # "logs/ablation.aug.ls50h.3x150/ls50h.a8.15.s0.1.p50.r25_p0.1lr0.0005.ls50h",
    ],
    # "ablation.aug.ls100h.3x200.ft": [
    #     "logs/ablation.aug.ls100h.3x200/ls100h.noauglr0.0005.ls100h",
    #     "logs/ablation.aug.ls100h.3x200/ls100h.add8.15lr0.0005.ls100h",
    #     # "logs/ablation.aug.ls100h.3x200/ls100h.sameaug.add8.15lr0.0005.ls100h",
    # ],
    # "ablation.aug.ls400h.3x300.ft": [
    #     "logs/ablation.aug.ls400h.3x300/ls400h.noauglr0.0005.ls400h",
    #     "logs/ablation.aug.ls400h.3x300/ls400h.add8.15lr0.0005.ls400h",
    # ],
    "ablation.aug.ls960h.3x400.ft": [
        "logs/ablation.aug.ls960h.3x400/ls960h.noauglr0.0005.ls960h",
        "logs/ablation.aug.ls960h.3x400/ls960h.add8.15lr0.0005.ls960h",
    ],
    # "ablation.mlp.ls960h.3x400.ft": [
    #     # "logs/ablation.mlp.ls960h.3x400/lr0.0005.cmlpFalse.tmlpTrue.bnFalse.actrelu.scale4.nhid0.unlab",
    #     # "logs/ablation.mlp.ls960h.3x400/lr0.0005.cmlpFalse.tmlpTrue.bnFalse.actrelu.scale4.nhid1.unlab",
    #     # "logs/ablation.mlp.ls960h.3x400/lr0.0005.cmlpFalse.tmlpTrue.bnFalse.actrelu.scale4.nhid2.unlab",
    #     "logs/ablation.mlp.ls960h.3x400/lr0.0005.cmlpTrue.tmlpTrue.bnTrue.actrelu.scale4.nhid0.unlab",
    #     "logs/ablation.mlp.ls960h.3x400/lr0.0005.cmlpTrue.tmlpTrue.bnTrue.actrelu.scale4.nhid1.unlab",
    #     "logs/ablation.mlp.ls960h.3x400/lr0.0005.cmlpTrue.tmlpTrue.bnTrue.actrelu.scale4.nhid2.unlab",
    # ],
    # # # # "ablation.glu.ls960h.3x400.ft": [
    # # # #     "logs/ablation.glu.ls960h.3x400/ls960h.geglu.lr0.0005.ls960h",
    # # # #     "logs/ablation.glu.ls960h.3x400/ls960h.swiglu.lr0.0005.ls960h",
    # # # #     "logs/ablation.glu.ls960h.3x400/ls960h.geglu_bias.lr0.0005.ls960h",
    # # # #     "logs/ablation.glu.ls960h.3x400/ls960h.swiglu_bias.lr0.0005.ls960h",
    # # # # ],
    # "ablation.lconv.ls960h.3x400.ft": [
    #     "logs/ablation.lconv.ls960h.3x400/ls960h.dc_last2.lr0.0005.moddefault.ls960h",
    #     "logs/ablation.lconv.ls960h.3x400/ls960h.dc_last2.lr0.0005.moddo0.1.ls960h",
    #     # "logs/ablation.lconv.ls960h.3x400/ls960h.dc_last4.lr0.0005.moddefault.ls960h",
    #     # "logs/ablation.lconv.ls960h.3x400/ls960h.dc_last4.lr0.0005.moddo0.1.ls960h",
    #     "logs/ablation.lconv.ls960h.3x400/ls960h.lc_last2.lr0.0005.moddefault.ls960h",
    #     "logs/ablation.lconv.ls960h.3x400/ls960h.lc_last2.lr0.0005.moddo0.1.ls960h",
    #     # "logs/ablation.lconv.ls960h.3x400/ls960h.lc_last4.lr0.0005.moddefault.ls960h",
    #     # "logs/ablation.lconv.ls960h.3x400/ls960h.lc_last4.lr0.0005.moddo0.1.ls960h",
    # ],
    # "ablation.conf.ls960h.3x400.ft": [
    #     "logs/ablation.conf.ls960h.3x400/ls960h.conf.lr0.0005.ks3.normbatchnorm.ls960h",
    #     # "logs/ablation.conf.ls960h.3x400/ls960h.conf.lr0.0005.ks3.normlayernorm.ls960h",
    #     # "logs/ablation.conf.ls960h.3x400/ls960h.conf_rp.lr0.0005.ks3.normbatchnorm.ls960h",
    #     "logs/ablation.conf.ls960h.3x400/ls960h.conf_rp.lr0.0005.ks3.normlayernorm.ls960h",
    # ],
    # "ablation.aug.cons.ls100h.3x200.ft": [
    #     "logs/ablation.aug.cons.ls100h.3x200/ls100h.aug.cons.add8.15lr0.0005.conscosinex0.001.ls100h",
    #     "logs/ablation.aug.cons.ls100h.3x200/ls100h.aug.cons.add8.15lr0.0005.conscosinex0.01.ls100h",
    #     "logs/ablation.aug.cons.ls100h.3x200/ls100h.aug.cons.add8.15lr0.0005.conscosinex0.1.ls100h",
    #     "logs/ablation.aug.cons.ls100h.3x200/ls100h.aug.cons.add8.15lr0.0005.conscosinex1.ls100h",
    #     "logs/ablation.aug.cons.ls100h.3x200/ls100h.aug.cons.add8.15lr0.0005.conscosinex10.ls100h",
    #     "logs/ablation.aug.cons.ls100h.3x200/ls100h.aug.cons.add8.15lr0.0005.consl1x0.001.ls100h",
    # ],
    # "combo.conf_mlp.ls960h.3x400.ft": [
    #     "logs/combo.conf_mlp.ls960h.3x400/ls960h.conf_mlp.lr0.0005.ks3.normbatchnorm.mlpmlpBoth.nhid2.ls960h",
    #     # "logs/combo.conf_mlp.ls960h.3x400/ls960h.conf_rp_mlp.lr0.0005.ks3.normlayernorm.mlpmlpBoth.nhid2.ls960h",
    # ],

    # "combo.conf_mlp_lconv.ls960h.3x400.ft": [
    #     "logs/combo.conf_mlp_lconv.ls960h.3x400/ls960h.conf_mlp_dc_last2.lr0.0005.ks3.normbatchnorm.mlpmlpBoth.nhid2.moddo0.1.ls960h",
    #     "logs/combo.conf_mlp_lconv.ls960h.3x400/ls960h.conf_mlp_lc_last2.lr0.0005.ks3.normbatchnorm.mlpmlpBoth.nhid2.moddo0.1.ls960h",
    #     # "logs/combo.conf_mlp_lconv.ls960h.3x400/ls960h.conf_rp_mlp_dc_last2.lr0.0005.ks3.normlayernorm.mlpmlpBoth.nhid2.moddo0.1.ls960h",
    #     # "logs/combo.conf_mlp_lconv.ls960h.3x400/ls960h.conf_rp_mlp_lc_last2.lr0.0005.ks3.normlayernorm.mlpmlpBoth.nhid2.moddo0.1.ls960h",
    # ],

    "combo.conf2_mlp.ls960h.3x400.ft": [
        # "logs/combo.conf2_mlp.ls960h.3x400/ls960h.conf_mlp.lr0.0005.ks3.normbatchnorm.mlpmlpBoth.nhid2.ls960h",
        # "logs/combo.conf2_mlp.ls960h.3x400/ls960h.conf_mlp.lr0.0005.ks3.normlayernorm.mlpmlpBoth.nhid2.ls960h",
        # "logs/combo.conf2_mlp.ls960h.3x400/ls960h.conf_rp_mlp.lr0.0005.ks3.normbatchnorm.mlpmlpBoth.nhid2.ls960h",
    ]
}


def get_best_wer(dir):
    wers = []
    for f in dir.glob("*.out"):
        with open(f) as f:
            for ln in f:
                if " | best_wer" in ln:
                    wer = float(ln.strip().split()[-1])
                    wers.append(wer)
    if len(wers) > 0:
        return min(wers)
    return None


@submit.register_sweep
def get_results(*args):
    run_args_list = {
        # "lab.1h",
        "lab.10h",
        # "lab.100h",
    }
    for name, checkpoints_list in checkpoints.items():
        for checkpoint in checkpoints_list:
            for dset in run_args_list:
                checkpoint = Path(checkpoint)
                # name += ".aug"
                job_name = name + "/" + checkpoint.name
                print(job_name)
                wers = {}
                # for dir in (Path("logs") / job_name).glob(f"{dset}*seed*"):
                for dir in (Path("logs") / job_name).glob(f"{dset}*"):
                    wer = get_best_wer(dir)
                    if "seed" in dir.name:
                        key = dir.name.split("seed")[0]
                    else:
                        key = dir.name.rsplit("lab", 1)[0]
                    wers[key] = wers.get(key, [])
                    wers[key].append(wer)

                for key, wer in wers.items():
                    if None in wer:
                        continue
                    wer = np.array(wer)
                    print(key, wer.shape, f"{wer.mean():.2f} +/- {2 * wer.std():.2f}", 
                          # wer.min(), wer.max(), 
                          sorted(wer))
                print()


@submit.register_sweep
def w2v_base(base_args):

    run_args_list = {
        # "lab.1h": dict(updates=20_000, nodes=1, gpus=1, update_freq=4),
        "lab.1h": dict(updates=25_000, nodes=1, gpus=1, update_freq=2),
        "lab.10h": dict(updates=30_000, nodes=1, gpus=1, update_freq=4),
        "lab.100h": dict(updates=100_000, nodes=1, gpus=1, update_freq=4),
    }
    seeds = {
        "lab.1h": [1],
        "lab.10h": [1],
        "lab.100h": [1],

        # "lab.1h": [2, 3, 4, 5],
        # "lab.10h": [2, 3],
        # "lab.100h": [2, 3],
    }

    default_params = {
        "lab.1h": {
            "validate-interval": 500,
            "save-interval": 100000,
            "save-interval-updates": 500,
            "keep-interval-updates": 1,
            "validate-after-updates": 5000,
            "mask-channel-prob": 0.25,
            "freeze-finetune-updates": 10_000,
        },
        "lab.10h": {
            "validate-interval": 500,
            "save-interval": 100000,
            "save-interval-updates": 500,
            "keep-interval-updates": 1,
            "validate-after-updates": 10000,
            "mask-channel-prob": 0.5,
            "freeze-finetune-updates": 10_000,
        },
        "lab.100h": {
            "validate-interval": 2000,
            "save-interval": 100000,
            "save-interval-updates": 2000,
            "keep-interval-updates": 1,
            "validate-after-updates": 10_000,
        }
    }

    mask_lens = [10]
    mask_probs = [
        0.45,
        0.65,
    ]
    dos = [0.1]
    lrs = [
        # 2e-05,
        3e-05,
        5e-05,
    ]

    for name, checkpoints_list in checkpoints.items():
        for checkpoint in checkpoints_list:
            for dset, run_params in run_args_list.items():
                checkpoint = Path(checkpoint)
                args = deepcopy(base_args)
                args.nodes = run_params["nodes"]
                args.gpus = run_params["gpus"]
                args.name = (args.name or name) + "/" + checkpoint.name

                param_sweeps = [
                    (
                        f"{dset}.lr{lr}.mlen{mlen}.mprob{mprob}.do{do}.seed{seed}",
                        {
                            # Checkpoint
                            "w2v-path": checkpoint / "checkpoint_best.pt",
                            # N-Gram
                            'wer-args': wer_args,

                            "lr": lr,
                            'mask-length': mlen,
                            'mask-prob': mprob,
                            "max-update": run_params["updates"],
                            "update-freq": run_params["update_freq"],
                            "phase-ratio": [0.1, 0.4, 0.5],

                            "augment-audio": False,
                            "layerdrop": do,
                            "activation-dropout": do,
                            "freeze-finetune-updates": 0,
                            "consistency-loss": False,
                            "seed": seed,

                            **default_params[dset]
                        }
                    )
                    for do in dos
                    for mlen in mask_lens
                    for mprob in mask_probs
                    for lr in lrs
                    for seed in seeds[dset]
                ]
                # submit.run_sweeps(args, base_params, param_sweeps, dataset=dset, skip_if_cp_exists=False)
                submit.run_sweeps(args, base_params, param_sweeps, dataset=dset, skip_if_cp_exists=False)


@submit.register_sweep
def w2v_base_aug(base_args):

    run_args_list = {
        # "lab.1h": dict(updates=20_000, nodes=1, gpus=1, update_freq=4),
        "lab.1h": dict(updates=25_000, nodes=1, gpus=1, update_freq=2),
        "lab.10h": dict(updates=30_000, nodes=1, gpus=1, update_freq=4),
        "lab.100h": dict(updates=100_000, nodes=1, gpus=1, update_freq=4),
    }
    seeds = {
        "lab.1h": [1],
        "lab.10h": [1],
        "lab.100h": [1],

        # "lab.1h": [2, 3, 4, 5],
        # "lab.10h": [2, 3],
        # "lab.100h": [2, 3],
    }

    default_params = {
        "lab.1h": {
            "validate-interval": 500,
            "save-interval": 100000,
            "save-interval-updates": 500,
            "keep-interval-updates": 1,
            "validate-after-updates": 5000,
            "mask-channel-prob": 0.25,
            "freeze-finetune-updates": 10_000,
        },
        "lab.10h": {
            "validate-interval": 500,
            "save-interval": 100000,
            "save-interval-updates": 500,
            "keep-interval-updates": 1,
            "validate-after-updates": 10000,
            "mask-channel-prob": 0.5,
            "freeze-finetune-updates": 10_000,
        },
        "lab.100h": {
            "validate-interval": 2000,
            "save-interval": 100000,
            "save-interval-updates": 2000,
            "keep-interval-updates": 1,
            "validate-after-updates": 10_000,
        }
    }

    mask_lens = [10]
    mask_probs = [
        0.30,
        0.45,
        0.65,
    ]
    dos = [0.1]
    lrs = [
        # 2e-05,
        3e-05,
        5e-05,
    ]

    for name, checkpoints_list in checkpoints.items():
        for checkpoint in checkpoints_list:
            for dset, run_params in run_args_list.items():
                checkpoint = Path(checkpoint)
                args = deepcopy(base_args)
                args.nodes = run_params["nodes"]
                args.gpus = run_params["gpus"]
                args.name = (args.name or name) + ".aug" + "/" + checkpoint.name

                param_sweeps = [
                    (
                        f"{dset}.lr{lr}.mlen{mlen}.mprob{mprob}.do{do}.seed{seed}",
                        {
                            # Checkpoint
                            "w2v-path": checkpoint / "checkpoint_best.pt",
                            # N-Gram
                            'wer-args': wer_args,

                            "lr": lr,
                            'mask-length': mlen,
                            'mask-prob': mprob,
                            "max-update": run_params["updates"],
                            "update-freq": run_params["update_freq"],
                            "phase-ratio": [0.1, 0.4, 0.5],

                            # "augment-audio": False,
                            "layerdrop": do,
                            "activation-dropout": do,
                            "freeze-finetune-updates": 0,
                            "consistency-loss": False,
                            "seed": seed,

                            "augment-audio": True,
                            "augmentations": "additive,pitch",
                            'augment-source-prob': 1.,
                            "snr-min": 8, "snr-max": 15, "pitch-shift-std": 50,

                            **default_params[dset]
                        }
                    )
                    for do in dos
                    for mlen in mask_lens
                    for mprob in mask_probs
                    for lr in lrs
                    for seed in seeds[dset]
                ]
                # submit.run_sweeps(args, base_params, param_sweeps, dataset=dset, skip_if_cp_exists=False)
                submit.run_sweeps(args, base_params, param_sweeps, dataset=dset, skip_if_cp_exists=False)


@submit.register_sweep
def w2v_base_search(base_args):

    run_args_list = {
        # "lab.1h": dict(updates=20_000, nodes=1, gpus=1, update_freq=4),
        "lab.10h": dict(updates=30_000, nodes=1, gpus=1, update_freq=4),
        # "lab.100h": dict(updates=100_000, nodes=1, gpus=1, update_freq=4),
    }
    seeds = {
        # "lab.1h": [1],
        # "lab.1h": [1, 2, 3, 4, 5],
        # # "lab.10h": [1, 2, 3, 4, 5],
        # "lab.10h": [1, 2, 3],
        "lab.10h": [1],
        # # "lab.10h": [2, 3],
        # "lab.100h": [1, 2],
    }

    default_params = {
        "lab.1h": {
            "validate-interval": 1000,
            "save-interval": 100000,
            "save-interval-updates": 1000,
            "keep-interval-updates": 1,
            "validate-after-updates": 10000,

            "mask-channel-prob": 0.25,
            "freeze-finetune-updates": 10_000,
        },
        "lab.10h": {
            "validate-interval": 500,
            "save-interval": 100000,
            "save-interval-updates": 500,
            "keep-interval-updates": 1,
            "validate-after-updates": 10000,
            "mask-channel-prob": 0.5,
            "freeze-finetune-updates": 10_000,
        },
        "lab.100h": {
            "validate-interval": 2000,
            "save-interval": 100000,
            "save-interval-updates": 2000,
            "keep-interval-updates": 1,
            "validate-after-updates": 10_000,
        }
    }

    mask_lens = [7, 10, 13]
    mask_probs = [
        0.45,
        0.55,
        0.65,
        0.75,
    ]
    dos = [0.1]
    lrs = [
        3e-05,
        5e-05,
    ]

    for name, checkpoints_list in checkpoints.items():
        for checkpoint in checkpoints_list:
            for dset, run_params in run_args_list.items():
                checkpoint = Path(checkpoint)
                args = deepcopy(base_args)
                args.nodes = run_params["nodes"]
                args.gpus = run_params["gpus"]
                args.name = (args.name or name) + "/" + checkpoint.name

                param_sweeps = [
                    (
                        f"{dset}.lr{lr}.mlen{mlen}.mprob{mprob}.do{do}.seed{seed}",
                        {
                            # Checkpoint
                            "w2v-path": checkpoint / "checkpoint_best.pt",
                            # N-Gram
                            'wer-args': wer_args,

                            "lr": lr,
                            'mask-length': mlen,
                            'mask-prob': mprob,
                            "max-update": run_params["updates"],
                            "update-freq": run_params["update_freq"],
                            "phase-ratio": [0.1, 0.4, 0.5],

                            "augment-audio": False,
                            "layerdrop": do,
                            "activation-dropout": do,
                            "freeze-finetune-updates": 0,
                            "consistency-loss": False,
                            "seed": seed,

                            **default_params[dset]
                        }
                    )
                    for do in dos
                    for mlen in mask_lens
                    for mprob in mask_probs
                    for lr in lrs
                    for seed in seeds[dset]
                ]
                # submit.run_sweeps(args, base_params, param_sweeps, dataset=dset, skip_if_cp_exists=False)
                submit.run_sweeps(args, base_params, param_sweeps, dataset=dset, skip_if_cp_exists=True)


@submit.register_sweep
def w2v_scratch(base_args):
    run_args_list = {
        # "lab.1h": dict(updates=20_000, nodes=1, gpus=1, update_freq=4),
        # "lab.10h": dict(updates=80_000, nodes=1, gpus=8, update_freq=1),
        "lab.100h": dict(updates=100_000, nodes=1, gpus=1, update_freq=4),
    }
    default_params = {
        "lab.1h": {
            "validate-interval": 500,
            "save-interval": 100000,
            "save-interval-updates": 500,
            "keep-interval-updates": 1,
            "validate-after-updates": 5000,

            "mask-channel-prob": 0.25,
            "freeze-finetune-updates": 10_000,
        },
        "lab.100h": {
            "validate-interval": 2000,
            "save-interval": 100000,
            "save-interval-updates": 2000,
            "keep-interval-updates": 1,
            "validate-after-updates": 10_000,
        }
    }
    mask_lens = [10]
    mask_probs = [
        0.45,
        0.65,
    ]
    dos = [0.1]
    lrs = [
        3e-04,
        3e-03,
    ]
    checkpoint = Path("logs/ablation.baseline.ls960h.8x400/ls960h.baselinelr0.0005.ls960h")
    name = "logs/ablation.scratch"
    for dset, run_params in run_args_list.items():
        args = deepcopy(base_args)
        args.nodes = run_params["nodes"]
        args.gpus = run_params["gpus"]
        args.name = (args.name or name)
        param_sweeps = [
            (
                f"{dset}.lr{lr}.mlen{mlen}.mprob{mprob}.do{do}",
                {
                    # Checkpoint
                    "w2v-path": checkpoint / "checkpoint_best.pt",
                    # N-Gram
                    'wer-args': wer_args,

                    "lr": lr,
                    'mask-length': mlen,
                    'mask-prob': mprob,
                    "max-update": run_params["updates"],
                    "update-freq": run_params["update_freq"],
                    "phase-ratio": [0.1, 0.4, 0.5],

                    "augment-audio": False,
                    "layerdrop": do,
                    "activation-dropout": do,
                    "freeze-finetune-updates": 0,
                    "feature-grad-mult": 1.,
                    "no-pretrained-weights": True,

                    **default_params[dset]
                }
            )
            for do in dos
            for mlen in mask_lens
            for mprob in mask_probs
            for lr in lrs
        ]
        submit.run_sweeps(args, base_params, param_sweeps, dataset=dset, skip_if_cp_exists=False)


if __name__ == '__main__':
    parser = submit.create_parser()
    base_args = parser.parse_args()
    sweep_func = submit.get_sweep_func(base_args.sweep)
    sweep_func(base_args)
