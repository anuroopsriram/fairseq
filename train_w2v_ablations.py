from copy import deepcopy
import submit


base_params = {
    'save-dir': '',
    'no-epoch-checkpoints': True,
    'fp16': True,
    'distributed-world-size': 1,
    'distributed-port': 13356,
    'num-workers': 10,
    'task': 'audio_pretraining',
    'criterion': 'wav2vec',
    'arch': 'wav2vec2',
    'log-keys': ["prob_perplexity", "code_perplexity", "temp"],
    'quantize-targets': True,
    'extractor-mode': 'default',
    'conv-feature-layers': [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2,
    'final-dim': 256,
    'latent-vars': 320,
    'latent-groups': 2,
    'latent-temp': (2, 0.5, 0.999995),
    'infonce': True,
    'optimizer': 'adam',
    'adam-betas': (0.9, 0.98),
    'adam-eps': 1e-06,
    'lr-scheduler': 'polynomial_decay',
    'total-num-update': 400000,
    'lr': 0.0005,
    'warmup-updates': 32000,
    'mask-length': 10,
    'mask-prob': 0.65,
    'mask-selection': 'static',
    'mask-other': 0,
    'encoder-layerdrop': 0.05,
    'dropout-input': 0.1,
    'dropout-features': 0.1,
    'feature-grad-mult': 0.1,
    'loss-weights': [0.1, 10],
    'conv-pos': 128,
    'conv-pos-groups': 16,
    'num-negatives': 100,
    'cross-sample-negatives': 0,
    'max-sample-size': 250000,
    'min-sample-size': 32000,
    'dropout': 0.1,
    'attention-dropout': 0.1,
    'weight-decay': 0.01,
    'max-update': 400000,
    'skip-invalid-size-inputs-valid-test': True,
    'update-freq': 1,
    'ddp-backend': 'no_c10d',

    'max-tokens': 2_800_000,
    # 'max-tokens': 4_000_000,
    # 'max-tokens': 1_400_000,
}


@submit.register_sweep
def w2v_baseline(base_args):
    lrs = [
        # 2e-4,
        5e-4,
    ]
    name = "baseline"
    run_args_list = {
        "ls960h": dict(name="ablation.baseline.ls960h.8x400", updates=400_000, nodes=4, update_freq=2),
    }

    for dset, run_args in run_args_list.items():
        param_sweeps = [
            (
                f"{dset}.{name}lr{lr}",
                {
                    "lr": lr,
                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": run_args["update_freq"],
                    'max-tokens': 1_400_000,
                },
            )
            for lr in lrs
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps, dataset=dset)


@submit.register_sweep
def w2v_base_augment_small(base_args):
    lrs = [
        # 2e-4,
        5e-4,
    ]
    augment_params = {
        # (augment, augmentations, augSrcProb, augTgtProb, augParams)
        "noaug": (False, "additive,speed", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.1}),
        "add8.15": (True, "additive", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.1}),
        # "add8.15_spd0.1": (True, "additive,speed", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.1}),
        # "add8.15_spd0.15": (True, "additive,speed", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.15}),
    }
    run_args_list = {
        "ls10h": dict(name="ablation.aug.50M.ls10h.3x100", updates=100_000, nodes=3, update_freq=1),
    }

    for dset, run_args in run_args_list.items():
        param_sweeps = [
            (
                f"{dset}.{name}lr{lr}",
                {
                    "lr": lr,
                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": 1,

                    # "encoder-layers": 8,
                    'conv-feature-layers': [(256, 10, 5)] + [(256, 3, 2)] * 4 + [(256, 2, 2)] * 2,
                    'encoder-embed-dim': 480,
                    
                    "augment-audio": augment,
                    "augmentations": augmentations,
                    'augment-source-prob': augSrcProb,
                    'augment-target-prob': augTgtProb,
                    **augParams
                },
            )
            for name, (augment, augmentations, augSrcProb, augTgtProb, augParams) in augment_params.items()
            for lr in lrs
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps, dataset=dset)


@submit.register_sweep
def w2v_base_augment(base_args):
    lrs = [
        # 2e-4,
        5e-4,
    ]
    augment_params = {
        # (augment, augmentations, augSrcProb, augTgtProb, augParams)
        "noaug": (False, "additive,speed", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.1}),
        "add8.15": (True, "additive", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.1}),
        # "add8.15_spd0.1": (True, "additive,speed", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.1}),
        # "add8.15_spd0.15": (True, "additive,speed", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.15}),
    }
    run_args_list = {
        "ls10h": dict(name="ablation.aug.ls10h.3x100", updates=100_000, nodes=3, update_freq=1),
        "ls50h": dict(name="ablation.aug.ls50h.3x150", updates=150_000, nodes=3, update_freq=1),
        # "ls100h": dict(name="ablation.aug.ls100h.3x200", updates=200_000, nodes=3, update_freq=1),
        # "ls400h": dict(name="ablation.aug.ls400h.3x300", updates=300_000, nodes=3, update_freq=1),
        # "ls960h": dict(name="ablation.aug.ls960h.3x400", updates=400_000, nodes=3, update_freq=1),
    }

    for dset, run_args in run_args_list.items():
        param_sweeps = [
            (
                f"{dset}.{name}lr{lr}",
                {
                    "lr": lr,
                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": 1,
                    
                    "augment-audio": augment,
                    "augmentations": augmentations,
                    'augment-source-prob': augSrcProb,
                    'augment-target-prob': augTgtProb,
                    **augParams
                },
            )
            for name, (augment, augmentations, augSrcProb, augTgtProb, augParams) in augment_params.items()
            for lr in lrs
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps, dataset=dset)


@submit.register_sweep
def w2v_base_same_augment(base_args):
    lrs = [
        # 2e-4,
        5e-4,
    ]
    augment_params = {
        # (augment, augmentations, augSrcProb, augTgtProb, augParams)
        "sameaug.add8.15": (True, "additive", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.1}),
    }
    run_args_list = {
        "ls100h": dict(name="ablation.aug.ls100h.3x200", updates=200_000, nodes=3, update_freq=1),
    }

    for dset, run_args in run_args_list.items():
        param_sweeps = [
            (
                f"{dset}.{name}lr{lr}",
                {
                    "lr": lr,
                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": 1,
                    
                    "augment-audio": augment,
                    "augmentations": augmentations,
                    'augment-source-prob': augSrcProb,
                    'augment-target-prob': augTgtProb,
                    "match-source-target-aug": True,

                    "max-tokens": 1_400_000,

                    **augParams
                },
            )
            for name, (augment, augmentations, augSrcProb, augTgtProb, augParams) in augment_params.items()
            for lr in lrs
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps, dataset=dset)


@submit.register_sweep
def w2v_base_augment_consistency(base_args):
    lrs = [
        # 2e-4,
        5e-4,
    ]
    augment_params = {
        # (augment, augmentations, augSrcProb, augTgtProb, augParams)
        "aug.cons.add8.15": (True, "additive", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.1}),
    }
    run_args_list = {
        "ls100h": dict(name="ablation.aug.cons.ls100h.3x200", updates=200_000, nodes=3, update_freq=1),
    }
    consistency_losses = {
        # "l1": [0.1, 1, 10],
        # # "l2": [0.0001],
        # "cosine": [0.1, 1, 10],
        
        "l1": [0.001, 0.01],
        "cosine": [0.001, 0.01],
    }
    for dset, run_args in run_args_list.items():
        param_sweeps = [
            (
                f"{dset}.{name}lr{lr}.cons{loss}x{weight}",
                {
                    "lr": lr,
                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": 1,

                    "consistency-loss": loss,
                    'loss-weights': [0.1, 10, weight],

                    "augment-audio": augment,
                    "augmentations": augmentations,
                    'augment-source-prob': augSrcProb,
                    'augment-target-prob': augTgtProb,
                    **augParams
                },
            )
            for name, (augment, augmentations, augSrcProb, augTgtProb, augParams) in augment_params.items()
            for lr in lrs
            for loss, weights in consistency_losses.items()
            for weight in weights
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps, dataset=dset)


@submit.register_sweep
def w2v_base_mlp(base_args):
    lrs = [
        # 2e-4,
        5e-4,
    ]
    mlp_params = {
        # (mlpContext, mlpTarget, BatchNorm, Scale, Activation)
        # (True, False, False, 4, "relu"),
        (False, True, False, 4, "relu"),
        (True, True, True, 4, "relu"),
    }
    mlp_nhiddens = [
        0,
        1,
        2,
    ]
    run_args_list = [
        dict(name="ablation.mlp.ls960h.3x400", updates=400_000, nodes=3, update_freq=1),
    ]
    for run_args in run_args_list:
        param_sweeps = [
            (
                f"lr{lr}.cmlp{contextMLP}.tmlp{targetMLP}.bn{batchnorm}.act{activation}.scale{scale}.nhid{nhidden}",
                {
                    "lr": lr,
                    "projection-mlp-context": contextMLP,
                    "target-mlp-context": targetMLP,
                    "mlp-batch-norm": batchnorm,
                    "mlp-scale": scale,
                    "mlp-activation": activation,
                    "mlp-nhidden": nhidden,
                    
                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": run_args["update_freq"],
                },
            )
            for contextMLP, targetMLP, batchnorm, scale, activation in mlp_params
            for lr in lrs
            for nhidden in mlp_nhiddens
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps)


@submit.register_sweep
def w2v_base_conf(base_args):
    lrs = [
        # 2e-4,
        5e-4,
    ]
    run_args_list = {
        "ls960h": dict(name="ablation.conf.ls960h.3x400", updates=400_000, nodes=3, update_freq=1),
    }

    num_layers = 14
    trans_types = {
        # "conf": ",".join(["conf" for _ in range(num_layers)]),
        "conf_rp": ",".join(["conf_relpos" for _ in range(num_layers)]),
    }
    norms = [
        "batchnorm",
        "layernorm",
    ]
    kern_sizes = [3]
    for dset, run_args in run_args_list.items():
        param_sweeps = [
            (
                f"{dset}.{name}.lr{lr}.ks{kern_sz}.norm{norm}",
                {
                    "lr": lr,
                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": run_args["update_freq"],
                    
                    # "max-tokens": 2_000_000,

                    "encoder-layers": num_layers,
                    "transformer-type": trans_type,
                    "conformer-kernel-size": kern_sz,
                    'encoder-embed-dim': 512,
                    # 'encoder-embed-dim': 448,
                    'encoder-attention-heads': 8,
                    "conformer-norm": norm,
                },
            )
            for lr in lrs
            for name, trans_type in trans_types.items()
            for kern_sz in kern_sizes
            for norm in norms
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps, dataset=dset)


@submit.register_sweep
def w2v_base_glu(base_args):
    lrs = [
        # 2e-4,
        5e-4,
    ]
    run_args_list = {
        "ls960h": dict(name="ablation.glu.ls960h.3x400", updates=400_000, nodes=3, update_freq=1),
    }
    glu_args = [
        # (activation, ffn_glu, fc_bias, name)
        # ("gelu", True, False, "geglu"),
        # ("swish", True, False, "swiglu"),
        ("gelu", True, True, "geglu_bias"),
        ("swish", True, True, "swiglu_bias"),
    ]
    for dset, run_args in run_args_list.items():
        param_sweeps = [
            (
                f"{dset}.{name}.lr{lr}",
                {
                    "lr": lr,
                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": run_args["update_freq"],

                    "activation-fn": act,
                    "ffn-glu": ffn_glu,
                    "no-fc-bias": not fc_bias,
                    "encoder-ffn-embed-dim": 2048,
                },
            )
            for lr in lrs
            for act, ffn_glu, fc_bias, name in glu_args
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps, dataset=dset)


@submit.register_sweep
def w2v_base_lconv(base_args):
    lrs = [
        # 2e-4,
        5e-4,
    ]
    run_args_list = {
        "ls960h": dict(name="ablation.lconv.ls960h.3x400", updates=400_000, nodes=3, update_freq=1),
    }
    # conv_feature_layers = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 3, 2)] * 2
    conv_feature_layers2 = [(640, 10, 5)] + [(640, 3, 2)] * 4 + [(640, 3, 2)] * 2
    conv_feature_layers4 = [(608, 10, 5)] + [(608, 3, 2)] * 4 + [(608, 3, 2)] * 2
    # conv_feature_layers = [(768, 10, 5)] + [(768, 3, 2)] * 4 + [(768, 3, 2)] * 2
    num_lyrs = len(conv_feature_layers2)
    conv_types = {
        "lc_last2":  (["conv"] * (num_lyrs-2) + ["light_conv"] * 2, conv_feature_layers2),
        "lc_last4":  (["conv"] * (num_lyrs-4) + ["light_conv"] * 4, conv_feature_layers4),
        "dc_last2":  (["conv"] * (num_lyrs-2) + ["dynamic_conv"] * 2, conv_feature_layers2),
        "dc_last4":  (["conv"] * (num_lyrs-4) + ["dynamic_conv"] * 4, conv_feature_layers4),
    }
    lconv_params = [
        # (encglu, wtsmax, wtdo, reldo, inpdo, normbef, attnhds, avgpool, mod)
        (True, True, 0.1, 0., 0., False, 8, False, "default"),
        (True, True, 0.1, 0.1, 0.1, False, 8, False, "do0.1"),
    ]

    for dset, run_args in run_args_list.items():
        param_sweeps = [
            (
                f"{dset}.{name}.lr{lr}.mod{mod}",
                {
                    "lr": lr,
                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": run_args["update_freq"],

                    "conv-feature-layers": conv_feat_lyrs,
                    "conv-types": ",".join(ctypes),

                    # "encoder-layers": 15,
                    "lconv-encoder-noglu": encglu,
                    "lconv-weight-nosoftmax": wtsmax,
                    "lconv-weight-dropout": wtdo,
                    "lconv-relu-dropout": reldo,
                    "lconv-input-dropout": inpdo,
                    "lconv-encoder-normalize-before": normbef,
                    "lconv-encoder-attention-heads": attnhds,
                    "lconv-avg-pool": avgpool,
                },
            )
            for lr in lrs
            for name, (ctypes, conv_feat_lyrs) in conv_types.items()
            for encglu, wtsmax, wtdo, reldo, inpdo, normbef, attnhds, avgpool, mod in lconv_params
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps, dataset=dset)


@submit.register_sweep
def w2v_base_augment_mlp(base_args):
    run_args_list = {
        "ls960h": dict(name="combinations.aug.mlp.ls960h.3x400", updates=400_000, nodes=3, update_freq=1),
    }
    lrs = [
        # 2e-4,
        5e-4,
    ]
    augment_params = {
        # (augment, augmentations, augSrcProb, augTgtProb, augParams)
        "add8.15": (True, "additive", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.1}),
    }
    mlp_params = {
        # (mlpContext, mlpTarget, BatchNorm, Scale, Activation)
        (True, True, True, 4, "relu"),
    }
    mlp_nhiddens = [
        2,
    ]
    for dset, run_args in run_args_list.items():
        param_sweeps = [
            (
                f"{dset}.{name}lr{lr}.mlpBoth.nhid{nhidden}",
                {
                    "lr": lr,
                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": run_args["update_freq"],
                    
                    "augment-audio": augment,
                    "augmentations": augmentations,
                    'augment-source-prob': augSrcProb,
                    'augment-target-prob': augTgtProb,
                    **augParams,

                    "projection-mlp-context": contextMLP,
                    "target-mlp-context": targetMLP,
                    "mlp-batch-norm": batchnorm,
                    "mlp-scale": scale,
                    "mlp-activation": activation,
                    "mlp-nhidden": nhidden,
                },
            )
            for name, (augment, augmentations, augSrcProb, augTgtProb, augParams) in augment_params.items()
            for contextMLP, targetMLP, batchnorm, scale, activation in mlp_params
            for nhidden in mlp_nhiddens
            for lr in lrs
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps, dataset=dset)


if __name__ == '__main__':
    parser = submit.create_parser()
    base_args = parser.parse_args()
    sweep_func = submit.get_sweep_func(base_args.sweep)
    sweep_func(base_args)
