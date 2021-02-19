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

    'criterion': 'siamese_wav2vec',
    'arch': 'siamese_wav2vec2',
    'log-keys': ["prob_perplexity", "code_perplexity", "temp"],

    'conv-feature-layers': [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2,
    'extractor-mode': 'default',
    'final-dim': 256,
    'latent-vars': 320,
    'latent-groups': 2,
    'latent-temp': (2, 0.5, 0.999995),
    
    'optimizer': 'adam',
    'adam-betas': (0.9, 0.98),
    'adam-eps': 1e-06,
    'lr-scheduler': 'polynomial_decay',
    'total-num-update': 400000,
    'lr': 0.0005,
    'warmup-updates': 32000,
    'weight-decay': 0.01,

    'encoder-layerdrop': 0.05,
    'dropout': 0.1,
    'dropout-features': 0.1,
    'attention-dropout': 0.1,
    'feature-grad-mult': 0.1,

    'mask-length': 10,
    'mask-prob': 0.65,
    'mask-selection': 'static',
    'mask-other': 0,

    'conv-pos': 128,
    'conv-pos-groups': 16,
    'max-sample-size': 250000,
    'min-sample-size': 32000,

    # 'max-tokens': 1400000,
    'max-tokens': 2000000,
    'max-update': 400000,
    'skip-invalid-size-inputs-valid-test': True,
    'ddp-backend no_c10d': True,
    'update-freq': 1,

    'quantize': False,
    'loss-weights': [10],
    'stop-gradient': True,

    'augment-audio': True,
    'augmentations': 'speed',
    'augment-source-prob': 1.,
    'augment-target-prob': 1.,
    'speed-std': 0.1,
}


@submit.register_sweep
def w2v_siamese(base_args):
    run_args_list = [
        dict(name="siamese.1x100", updates=100_000, nodes=1, update_freq=1),
    ]

    lrs = [
        5e-5,
        2e-4,
        5e-4,
        2e-3,
    ]
    mask_probs = [
        # 0.,
        # 0.25,
        # 0.5,
        0.65,
    ]
    augment_params = [
        # (augment, augmentations, augSrcProb, augTgtProb, augParams)
        # (True, "speed", 1., 1., {"speed-std": 0.05}),
        # (True, "speed", 1., 1., {"speed-std": 0.10}),
        # (True, "speed", 1., 1., {"speed-std": 0.20}),

        # (True, "additive,speed", 1., 1., {"speed-std": 0.1, "snr-min": 10, "snr-max": 15,}),
        # (True, "additive,speed", 1., 1., {"speed-std": 0.2, "snr-min": 8, "snr-max": 15,}),
        # (True, "additive,speed", 1., 1., {"speed-std": 0.3, "snr-min": 5, "snr-max": 15,}),
        (True, "additive,pitch,speed,reverb", 1., 1., {"speed-std": 0.15, "snr-min": 5, "snr-max": 15, "pitch-shift-std": 200}),
    ]
    for run_args in run_args_list:
        param_sweeps = [
            (
                f"lr{lr}.mp{mp}" + 
                "_".join(f"{key}{val}" for key, val in augParams.items()),
                {
                    "lr": lr,
                    "mask-prob": mp,
                    **augParams,

                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": run_args["update_freq"],
                },
            )
            for augment, augmentations, augSrcProb, augTgtProb, augParams in augment_params
            for lr in lrs
            for mp in mask_probs
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
