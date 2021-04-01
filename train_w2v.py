from copy import deepcopy
import submit


base_params = {
    'save-dir': '',
    'no-epoch-checkpoints': True,
    'fp16': True,
    'distributed-world-size': 1,
    'distributed-port': 13356,
    'num-workers': 0,
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
    'max-tokens': 1400000,
    'max-update': 400000,
    'skip-invalid-size-inputs-valid-test': True,
    'ddp-backend': 'no_c10d',
    'update-freq': 1,
}


@submit.register_sweep
def w2v_base(base_args):
    lrs = [5e-4]
    run_args_list = [
        dict(name="w2v.base.2x100", updates=100_000, nodes=2, update_freq=1),
    ]
    for run_args in run_args_list:
        param_sweeps = [
            (
                f"lr{lr}",
                {
                    "lr": lr,
                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": run_args["update_freq"],

                    # "ddp-backend": "fully_sharded",
                    # "cpu-offload": True,
                    # "no-reshard-after-forward": True,
                },
            )
            for lr in lrs
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps)


@submit.register_sweep
def w2v_base_conformer(base_args):
    lrs = [5e-4]
    run_args_list = [
        dict(name="w2v.base.conf.2x100", updates=100_000, nodes=2, update_freq=1),
    ]
    num_layers = 17
    trans_types = {
        "conf": ",".join(["conf" for _ in range(num_layers)]),
        "conf_rp": ",".join(["conf_relpos" for _ in range(num_layers)]),
    }
    norms = [
        # "batchnorm",
        "layernorm",
    ]
    kern_sizes = [3]
    for run_args in run_args_list:
        param_sweeps = [
            (
                f"lr{lr}.trans{trans}.norm{norm}",
                {
                    "lr": lr,
                    "encoder-layers": num_layers,
                    "transformer-type": trans_type,
                    "conformer-kernel-size": kern_sz,
                    'encoder-embed-dim': 512,
                    'encoder-attention-heads': 8,
                    "conformer-norm": norm,

                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": run_args["update_freq"],
                },
            )
            for norm in norms
            for lr in lrs
            for kern_sz in kern_sizes
            for trans, trans_type in trans_types.items()
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps)


@submit.register_sweep
def w2v_base_lightconv(base_args):
    run_args_list = [
        dict(name="w2v.base.lc.2x100", updates=100_000, nodes=2, update_freq=1),
        # dict(name="w2v.base.lc.8x400", updates=400_000, nodes=8, update_freq=1),
    ]
    lrs = [5e-4]
    # kernels = [3, 5, 7, 15, 31, 63, 127, 255]
    # conv_feature_layers = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
    conv_feature_layers = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 3, 2)] * 2
    num_lyrs = len(conv_feature_layers)
    conv_types = {
        # "conv": ["conv"] * num_lyrs,
        "lc_last2":  ["conv"] * (num_lyrs-2) + ["light_conv"] * 2,
        # "lc_last4":  ["conv"] * (num_lyrs-4) + ["light_conv"] * 4,
        # "lc_last6":  ["conv"] * (num_lyrs-6) + ["light_conv"] * 6,
        # "dc_last2":  ["conv"] * (num_lyrs-2) + ["dynamic_conv"] * 2,
        # "dc_last4":  ["conv"] * (num_lyrs-4) + ["dynamic_conv"] * 4,
        # "dc_last6":  ["conv"] * (num_lyrs-6) + ["dynamic_conv"] * 6,
        # "conv": ["conv"] * num_lyrs,
    }
    lconv_params = [
        # (lconv_encoder_glu, lconv_weight_softmax, lconv_weight_dropout, lconv_relu_dropout, lconv_input_dropout, lconv_encoder_normalize_before, lconv_encoder_attention_heads, lconv_avg_pool)
        (True, True, 0.1, 0., 0., False, 2, True, "None"),  # Default

        # (False, True, 0.1, 0., 0., False, 2, True, "noglu"),  # No encoder glu
        # (True, False, 0.1, 0., 0., False, 2, True, "nowtsmax"),  # No weight softmax
        # (True, True, 0.0, 0., 0., False, 2, True, "nodo"),  # No dropout
        # (True, True, 0.1, 0., 0., True, 2, True, "nonormbef"),  # Normalize before
        # (True, True, 0.1, 0., 0., False, 4, True, "4hds"),  # 4 heads
        # (True, True, 0.1, 0., 0., False, 8, True, "8hds"),  # 8 heads
        # (True, True, 0.1, 0., 0., False, 2, False, "noavgpool"),  # No avg pool
    ]

    for run_args in run_args_list:
        param_sweeps = [
            (
                f"lr{lr}.ctype_{conv_name}.lconvparams{lconvrepr}",
                {
                    "lr": lr,
                    "conv-types": ",".join(ctypes),
                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": run_args["update_freq"],
                    'conv-feature-layers': conv_feature_layers,

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
            for conv_name, ctypes in conv_types.items()
            for encglu, wtsmax, wtdo, reldo, inpdo, normbef, attnhds, avgpool, lconvrepr in lconv_params
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps)


@submit.register_sweep
def w2v_base_mlp(base_args):
    lrs = [5e-4]
    mlp_params = {
        # (mlpContext, mlpTarget, BatchNorm, Scale, Activation)
        # (False, False, False, 1, "relu"),
        # (True, False, False, 4, "relu"),
        # (False, True, False, 4, "relu"),
        # (False, False, True, 4, "relu"),
        (True, True, True, 4, "relu"),
    }
    run_args_list = [
        # dict(name="w2v.base.mlp.2x100", updates=100_000, nodes=2, update_freq=1),
        # dict(name="w2v.base.mlp.4x400", updates=400_000, nodes=4, update_freq=1),
        dict(name="w2v.base.mlp.8x400", updates=400_000, nodes=4, update_freq=2),
    ]
    for run_args in run_args_list:
        param_sweeps = [
            (
                f"lr{lr}.contextmlp{contextMLP}.tgtmlp{targetMLP}.bn{batchnorm}.act{activation}.scale{scale}",
                {
                    "lr": lr,
                    "projection-mlp-context": contextMLP,
                    "target-mlp-context": targetMLP,
                    "mlp-batch-norm": batchnorm,
                    "mlp-scale": scale,
                    "mlp-activation": activation,
                    
                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": run_args["update_freq"],
                },
            )
            for contextMLP, targetMLP, batchnorm, scale, activation in mlp_params
            for lr in lrs
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps)


@submit.register_sweep
def w2v_base_augment(base_args):
    lrs = [5e-4]
    augment_params = [
        # (augment, augmentations, augSrcProb, augTgtProb, augParams)
        # (True, "additive", 1., 1., {"snr-min": 5, "snr-max": 15}),
        # (True, "additive", 1., 1., {"snr-min": 8, "snr-max": 15}),
        # (True, "additive", 1., 1., {"snr-min": 10, "snr-max": 15}),
        (True, "speed", 1., 1., {"speed-std": 0.05}),
        (True, "speed", 1., 1., {"speed-std": 0.1}),
        (True, "speed", 1., 1., {"speed-std": 0.15}),
    ]
    drop_params = [
        # (dropout, layerdrop)
        (0., 0.),
        # (0., 0.05),
        # (0.1, 0.),
        (0.1, 0.05),
    ]
    normalize = [False, True]
    run_args_list = [
        dict(name="w2v.base.augment.2x100", updates=100_000, nodes=2),
    ]
    for run_args in run_args_list:
        param_sweeps = [
            (
                (
                    f"lr{lr}.do{do}.ld{ld}.norm{norm}" +
                    f"augSrc{augSrcProb}.augTgt{augTgtProb}.augs{augmentations}." +
                    "_".join(f"{key}{val}" for key, val in augParams.items())
                ),
                {
                    "lr": lr,
                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": 1,
                    
                    "augment-audio": augment,
                    "augmentations": augmentations,
                    'augment-source-prob': augSrcProb,
                    'augment-target-prob': augTgtProb,
                    "normalize": norm,

                    "dropout-input": do,
                    "dropout-features": do,
                    "dropout": do,
                    "encoder-layerdrop": ld,

                    **augParams
                },
            )
            for augment, augmentations, augSrcProb, augTgtProb, augParams in augment_params
            for lr in lrs
            for do, ld in drop_params
            for norm in normalize
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps)


@submit.register_sweep
def w2v_base_augment_100h(base_args):
    lrs = [5e-4]
    augment_params = [
        # (augment, augmentations, augSrcProb, augTgtProb, augParams, name)
        (False, "additive", 0., 0., {"speed-std": 0.}, "baseline"),
        (True, "additive", 1., 1., {"snr-min": 8, "snr-max": 15}, "add_8_15"),
        (True, "speed", 1., 1., {"speed-std": 0.1}, "spd0.1"),
    ]
    drop_params = [
        # (dropout, layerdrop)
        (0., 0.),
        (0.1, 0.05),
    ]
    run_args_list = [
        dict(name="w2v.base.augment.2x100.100h", updates=100_000, nodes=2),
    ]
    for run_args in run_args_list:
        param_sweeps = [
            (
                (
                    f"{name}.lr{lr}.do{do}.ld{ld}"
                ),
                {
                    "lr": lr,
                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": 1,
                    
                    "augment-audio": augment,
                    "augmentations": augmentations,
                    'augment-source-prob': augSrcProb,
                    'augment-target-prob': augTgtProb,
                    "normalize": False,

                    "dropout-input": do,
                    "dropout-features": do,
                    "dropout": do,
                    "encoder-layerdrop": ld,

                    **augParams
                },
            )
            for augment, augmentations, augSrcProb, augTgtProb, augParams, name in augment_params
            for lr in lrs
            for do, ld in drop_params
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps, dataset='unlab.100')


@submit.register_sweep
def w2v_base_mlp_augment_aug_search(base_args):
    lrs = [5e-4]
    augment_params = [
        # (augment, augmentations, augSrcProb, augTgtProb, augParams)
        (True, "additive,speed", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.1}),
        (True, "additive,speed", 1., 1., {"snr-min": 5, "snr-max": 15, "speed-std": 0.1}),
        (True, "additive,speed", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.15}),
        (True, "additive,speed,pitch", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.1, "pitch-shift-std": 10}),
        (True, "additive,speed,reverb", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.1, "reverb-strength": 30}),
    ]
    run_args_list = [
        dict(name="w2v.base.mlp.augment.search.4x400", updates=400_000, nodes=4, update_freq=1),
    ]
    for run_args in run_args_list:
        param_sweeps = [
            (
                (
                    f"lr{lr}.tgtmlp." +
                    f"augSrc{augSrcProb}.augTgt{augTgtProb}.augs{augmentations}." +
                    "_".join(f"{key}{val}" for key, val in augParams.items())
                ),
                {
                    "lr": lr,
                    "projection-mlp-context": False,
                    "target-mlp-context": True,
                    "mlp-batch-norm": True,
                    "mlp-scale": 4,
                    "mlp-activation": "relu",

                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": run_args["update_freq"],

                    "augment-audio": augment,
                    "augmentations": augmentations,
                    'augment-source-prob': augSrcProb,
                    'augment-target-prob': augTgtProb,
                    "normalize": False,

                    "dropout-input": 0.1,
                    "dropout-features": 0.1,
                    "dropout": 0.1,
                    "encoder-layerdrop": 0.05,

                    **augParams
                },
            )
            for augment, augmentations, augSrcProb, augTgtProb, augParams in augment_params
            for lr in lrs
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps)


@submit.register_sweep
def w2v_base_mlp_augment_ablation(base_args):
    lrs = [5e-4]
    mlp_params = {
        # (mlpContext, mlpTarget, BatchNorm, Scale, Activation)
        (False, False, False, 1, "relu"),
        (False, True, True, 4, "relu"),
        (True, False, True, 4, "relu"),
        (True, True, True, 4, "relu"),
    }
    augment_params = [
        # (augment, augmentations, augSrcProb, augTgtProb, augParams)
        (False, "additive", 0., 0., {"speed-std": 0.1}),
        (True, "additive,speed", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.1}),
    ]
    run_args_list = [
        dict(name="w2v.base.mlp.augment.4x400", updates=400_000, nodes=4, update_freq=1),
    ]
    drop_params = [
        # (dropout, layerdrop)
        (0.1, 0.05),
    ]
    for run_args in run_args_list:
        param_sweeps = [
            (
                (
                    f"lr{lr}.contextmlp{contextMLP}.tgtmlp{targetMLP}.bn{batchnorm}.act{activation}.scale{scale}.do{do}.ld{ld}.normFalse" +
                    f"augSrc{augSrcProb}.augTgt{augTgtProb}.augs{augmentations}." +
                    "_".join(f"{key}{val}" for key, val in augParams.items())
                ),
                {
                    "lr": lr,
                    "projection-mlp-context": contextMLP,
                    "target-mlp-context": targetMLP,
                    "mlp-batch-norm": batchnorm,
                    "mlp-scale": scale,
                    "mlp-activation": activation,

                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": run_args["update_freq"],

                    "augment-audio": augment,
                    "augmentations": augmentations,
                    'augment-source-prob': augSrcProb,
                    'augment-target-prob': augTgtProb,
                    "normalize": False,

                    "dropout-input": do,
                    "dropout-features": do,
                    "dropout": do,
                    "encoder-layerdrop": ld,

                    **augParams
                },
            )
            for contextMLP, targetMLP, batchnorm, scale, activation in mlp_params
            for augment, augmentations, augSrcProb, augTgtProb, augParams in augment_params
            for lr in lrs
            for do, ld in drop_params
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps)


@submit.register_sweep
def w2v_base_mlp_augment_conf_ablation(base_args):
    lrs = [5e-4]
    mlp_params = {
        # (mlpContext, mlpTarget, BatchNorm, Scale, Activation)
        (False, True, True, 4, "relu"),
    }
    augment_params = [
        # (augment, augmentations, augSrcProb, augTgtProb, augParams)
        (True, "additive,speed", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.1}),
    ]
    run_args_list = [
        dict(name="w2v.base.mlp.augment.4x400", updates=400_000, nodes=4, update_freq=1),
    ]
    num_layers = 17
    trans_types = {
        "conf": ",".join(["conf" for _ in range(num_layers)]),
        "conf_rp": ",".join(["conf_relpos" for _ in range(num_layers)]),
    }
    drop_params = [
        # (dropout, layerdrop)
        (0.1, 0.05),
    ]
    for run_args in run_args_list:
        param_sweeps = [
            (
                (
                    f"lr{lr}.trans{trans}.contextmlp{contextMLP}.tgtmlp{targetMLP}.bn{batchnorm}.act{activation}.scale{scale}.do{do}.ld{ld}.normFalse" +
                    f"augSrc{augSrcProb}.augTgt{augTgtProb}.augs{augmentations}." +
                    "_".join(f"{key}{val}" for key, val in augParams.items())
                ),
                {
                    "lr": lr,
                    "projection-mlp-context": contextMLP,
                    "target-mlp-context": targetMLP,
                    "mlp-batch-norm": batchnorm,
                    "mlp-scale": scale,
                    "mlp-activation": activation,

                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": run_args["update_freq"],

                    "augment-audio": augment,
                    "augmentations": augmentations,
                    'augment-source-prob': augSrcProb,
                    'augment-target-prob': augTgtProb,
                    "normalize": False,

                    "dropout-input": do,
                    "dropout-features": do,
                    "dropout": do,
                    "encoder-layerdrop": ld,

                    **augParams
                },
            )
            for contextMLP, targetMLP, batchnorm, scale, activation in mlp_params
            for augment, augmentations, augSrcProb, augTgtProb, augParams in augment_params
            for lr in lrs
            for do, ld in drop_params
            for trans, trans_type in trans_types.items()
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps)


@submit.register_sweep
def w2v_base_mlp_augment(base_args):
    lrs = [5e-4]
    # lrs = [2e-4]
    mlp_params = {
        # (mlpContext, mlpTarget, BatchNorm, Scale, Activation)
        # (False, False, False, 1, "relu"),

        # (False, True, True, 4, "relu"),
        # (True, False, True, 4, "relu"),
        (True, True, True, 4, "relu"),
    }
    
    augment_params = [
        # (augment, augmentations, augSrcProb, augTgtProb, augParams)
        # (True, "additive", 1., 1., {"snr-min": 5, "snr-max": 15}),

        # (True, "additive", 0., 1., {"snr-min": 8, "snr-max": 15}),
        # (True, "additive", 0., 1., {"snr-min": 10, "snr-max": 15}),
        # (True, "speed", 0., 1., {"speed-std": 0.05}),
        # (True, "speed", 0., 1., {"speed-std": 0.1}),
        # (True, "speed", 0., 1., {"speed-std": 0.15}),
        # (True, "additive,speed", 0., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.1}),
        # (True, "additive,speed", 0., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.15}),

        # (True, "additive", 1., 0., {"snr-min": 8, "snr-max": 15}),
        # (True, "additive", 1., 0., {"snr-min": 10, "snr-max": 15}),
        # (True, "speed", 1., 0., {"speed-std": 0.05}),
        # (True, "speed", 1., 0., {"speed-std": 0.1}),
        # (True, "speed", 1., 0., {"speed-std": 0.15}),
        # (True, "additive,speed", 1., 0., {"snr-min": 8, "snr-max": 15, "speed-std": 0.1}),
        # (True, "additive,speed", 1., 0., {"snr-min": 8, "snr-max": 15, "speed-std": 0.15}),

        # (True, "additive", 1., 1., {"snr-min": 8, "snr-max": 15}),
        # (True, "additive", 1., 1., {"snr-min": 10, "snr-max": 15}),
        # (True, "speed", 1., 1., {"speed-std": 0.05}),
        # (True, "speed", 1., 1., {"speed-std": 0.1}),
        # (True, "speed", 1., 1., {"speed-std": 0.15}),
        (True, "additive,speed", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.1}),
        # (True, "additive,speed", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.15}),
        # (True, "additive,speed", 1., 1., {"snr-min": 6, "snr-max": 15, "speed-std": 0.15}),
        # (True, "additive,speed", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.20}),
        # (True, "additive,speed", 1., 1., {"snr-min": 10, "snr-max": 15, "speed-std": 0.1}),

        # (True, "pitch", 1., 1., {"pitch-shift-std": 10}),
        # (True, "pitch", 1., 1., {"pitch-shift-std": 50}),
        # (True, "pitch", 1., 1., {"pitch-shift-std": 100}),
    ]
    run_args_list = [
        # dict(name="w2v.base.mlp.augment.2x100", updates=100_000, nodes=2),
        # dict(name="w2v.base.mlp.augment.4x400", updates=400_000, nodes=4),
        # dict(name="w2v.base.mlp.augment.8x400", updates=400_000, nodes=4, update_freq=2),
        dict(name="w2v.base.mlp.augment.8x400", updates=400_000, nodes=8, update_freq=1),
    ]
    drop_params = [
        # (dropout, layerdrop)
        # (0., 0.),
        # (0.05, 0.025),
        (0.1, 0.05),
    ]
    for run_args in run_args_list:
        param_sweeps = [
            (
                (
                    f"lr{lr}.contextmlp{contextMLP}.tgtmlp{targetMLP}.bn{batchnorm}.act{activation}.scale{scale}.do{do}.ld{ld}.normFalse" +
                    f"augSrc{augSrcProb}.augTgt{augTgtProb}.augs{augmentations}." +
                    "_".join(f"{key}{val}" for key, val in augParams.items())
                ),
                {
                    "lr": lr,
                    "projection-mlp-context": contextMLP,
                    "target-mlp-context": targetMLP,
                    "mlp-batch-norm": batchnorm,
                    "mlp-scale": scale,
                    "mlp-activation": activation,

                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": run_args["update_freq"],

                    "augment-audio": augment,
                    "augmentations": augmentations,
                    'augment-source-prob': augSrcProb,
                    'augment-target-prob': augTgtProb,
                    # "normalize": True,
                    "normalize": False,

                    "dropout-input": do,
                    "dropout-features": do,
                    "dropout": do,
                    "encoder-layerdrop": ld,

                    **augParams
                },
            )
            for contextMLP, targetMLP, batchnorm, scale, activation in mlp_params
            for augment, augmentations, augSrcProb, augTgtProb, augParams in augment_params
            for lr in lrs
            for do, ld in drop_params
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps)


@submit.register_sweep
def w2v_base_conformer_mlp_augment(base_args):
    lrs = [5e-4]
    run_args_list = [
        # dict(name="w2v.base.conf.2x100", updates=100_000, nodes=2, update_freq=1),
        dict(name="w2v.base.conf.8x400", updates=400_000, nodes=8, update_freq=1),
    ]
    num_layers = 17
    trans_types = {
        "conf": ",".join(["conf" for _ in range(num_layers)]),
        "conf_rp": ",".join(["conf_relpos" for _ in range(num_layers)]),
    }
    mlp_params = [
        # (mlpContext, mlpTarget, BatchNorm, Scale, Activation)
        (False, True, True, 4, "relu"),
        # (True, True, True, 4, "relu"),
    ]
    augment_params = [
        # (augment, augmentations, augSrcProb, augTgtProb, augParams)
        # (False, "speed", 1., 1., {"speed-std": 0.}),
        (True, "additive,speed", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.1}),
        # (True, "additive,speed", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.15}),
    ]
    drop_params = [
        # (dropout, layerdrop)
        # (0.05, 0.025),
        (0.1, 0.05),
        # (0., 0.),
    ]
    kern_sizes = [3]
    for run_args in run_args_list:
        param_sweeps = [
            (
                (
                    f"lr{lr}.trans{trans}.ksz{kern_sz}.cmlp{contextMLP}.tmlp{targetMLP}.bn{batchnorm}.act{activation}.scale{scale}.do{do}.ld{ld}.normFalse" +
                    f"augSrc{augSrcProb}.augTgt{augTgtProb}" + 
                    (".augs{augmentations}." + "_".join(f"{key}{val}" for key, val in augParams.items()) if augmentations else "")
                ),
                {
                    "lr": lr,
                    "encoder-layers": num_layers,
                    "transformer-type": trans_type,
                    "conformer-kernel-size": kern_sz,
                    'encoder-embed-dim': 512,
                    'encoder-attention-heads': 8,

                    "projection-mlp-context": contextMLP,
                    "target-mlp-context": targetMLP,
                    "mlp-batch-norm": batchnorm,
                    "mlp-scale": scale,
                    "mlp-activation": activation,

                    "augment-audio": augment,
                    "augmentations": augmentations,
                    'augment-source-prob': augSrcProb,
                    'augment-target-prob': augTgtProb,
                    "normalize": False,

                    "dropout-input": do,
                    "dropout-features": do,
                    "dropout": do,
                    "encoder-layerdrop": ld,

                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": run_args["update_freq"],
                },
            )
            for contextMLP, targetMLP, batchnorm, scale, activation in mlp_params
            for augment, augmentations, augSrcProb, augTgtProb, augParams in augment_params
            for lr in lrs
            for do, ld in drop_params
            for kern_sz in kern_sizes
            for trans, trans_type in trans_types.items()
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps)


@submit.register_sweep
def w2v_base_glu(base_args):
    lrs = [
        2e-4,
        5e-4,
        # 1e-3,
    ]
    run_args_list = [
        # dict(name="w2v.base.glu.2x100", updates=100_000, nodes=2, update_freq=1),
        dict(name="w2v.base.glu.4x400", updates=400_000, nodes=4, update_freq=1),
    ]
    glu_args = [
        # (activation, ffn_glu, fc_bias, name)
        # ("gelu", True, False, "geglu"),
        ("gelu", True, True, "geglu_bias"),
        # ("swish", True, False, "swish"),
        ("swish", True, True, "swish_bias"),
    ]
    for run_args in run_args_list:
        param_sweeps = [
            (
                f"{name}.lr{lr}",
                {
                    "lr": lr,
                    "activation-fn": act,
                    "ffn-glu": ffn_glu,
                    "no-fc-bias": not fc_bias,
                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": run_args["update_freq"],
                    "encoder-ffn-embed-dim": 2048,
                },
            )
            for lr in lrs
            for act, ffn_glu, fc_bias, name in glu_args
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps)


@submit.register_sweep
def w2v_base_agc(base_args):
    lrs = [5e-4]
    run_args_list = [
        dict(name="w2v.base.agc.2x100", updates=100_000, nodes=2, update_freq=1),
    ]
    norm_params = [
        # (agc, nonorm, name)
        # (False, False, "baseline"),
        # (True, False, "agc_norm"),
        # (True, True, "agc_nonormconv"),
        (True, True, "agc_nonorm"),
    ]
    agc_clips = [
        # 3e-3, 1e-2, 3e-2
        1e-4, 1e-3, 1e-1, 3e-1,
    ]
    agc_eps = [
        # 3e-4, 1e-3, 3e-3
        3e-5, 1e-4, 1e-2, 3e-2
    ]
    for run_args in run_args_list:
        param_sweeps = [
            (
                f"{name}.lr{lr}.clip{clip}.eps{eps}",
                {
                    "lr": lr,
                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": run_args["update_freq"],
                    "agc": agc,
                    "nonorm": nonorm,
                    "agc-clipping": clip,
                    "agc-eps": eps,
                },
            )
            for agc, nonorm, name in norm_params
            for lr in lrs
            for clip in agc_clips
            for eps in agc_eps
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps)


@submit.register_sweep
def w2v_base_consistency(base_args):
    lrs = [5e-4]
    run_args_list = [
        dict(name="w2v.base.consistency.2x100", updates=100_000, nodes=2, update_freq=1),
    ]
    consistency_losses = [
        "l1",
        "l2",
        # "cosine"
    ]
    consistency_loss_weights = [
        0.00001,
        0.0001,
        0.001
        # 0.01,
        # 0.1,
        # 1.,
        # 10.,
        # 100.,
    ]
    for run_args in run_args_list:
        param_sweeps = [
            (
                f"lr{lr}.consistency{loss}x{weight}",
                {
                    "lr": lr,
                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": run_args["update_freq"],
                    
                    "consistency-loss": loss,
                    'loss-weights': [0.1, 10, weight],

                    "augment-audio": True,
                    'augment-source-prob': 1.,
                    'augment-target-prob': 1.,
                    "augmentations": "additive",
                    "snr-min": 8,
                    "snr-max": 15,
                    # "speed-std": 0.1,
                    # "augmentations": "speed",
                },
            )
            for lr in lrs
            for loss in consistency_losses
            for weight in consistency_loss_weights
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps)


if __name__ == '__main__':
    parser = submit.create_parser()
    base_args = parser.parse_args()
    sweep_func = submit.get_sweep_func(base_args.sweep)
    sweep_func(base_args)
