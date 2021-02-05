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
    'ddp-backend no_c10d': True,
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
                },
            )
            for lr in lrs
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
def w2v_base_mlp_augment(base_args):
    lrs = [5e-4]
    mlp_params = {
        # (mlpContext, mlpTarget, BatchNorm, Scale, Activation)
        # (False, False, False, 1, "relu"),
        (False, True, True, 4, "relu"),
        (True, False, True, 4, "relu"),
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
        (True, "additive,speed", 1., 1., {"snr-min": 8, "snr-max": 15, "speed-std": 0.15}),

        # (True, "pitch", 1., 1., {"pitch-shift-std": 10}),
        # (True, "pitch", 1., 1., {"pitch-shift-std": 50}),
        # (True, "pitch", 1., 1., {"pitch-shift-std": 100}),
    ]
    run_args_list = [
        dict(name="w2v.base.mlp.augment.2x100", updates=100_000, nodes=2),
        # dict(name="w2v.base.mlp.augment.4x400", updates=400_000, nodes=4),
        # dict(name="w2v.base.mlp.augment.8x400", updates=400_000, nodes=4, update_freq=2),
    ]
    drop_params = [
        # (dropout, layerdrop)
        (0., 0.),
        (0.05, 0.025),
        (0.1, 0.05),
    ]
    for run_args in run_args_list:
        param_sweeps = [
            (
                (
                    f"lr{lr}.contextmlp{contextMLP}.tgtmlp{targetMLP}.bn{batchnorm}.act{activation}.scale{scale}.do{do}.ld{ld}" +
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
                    "update-freq": 1,

                    "augment-audio": augment,
                    "augmentations": augmentations,
                    'augment-source-prob': augSrcProb,
                    'augment-target-prob': augTgtProb,
                    "normalize": True,

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


if __name__ == '__main__':
    parser = submit.create_parser()
    base_args = parser.parse_args()
    sweep_func = submit.get_sweep_func(base_args.sweep)
    sweep_func(base_args)
