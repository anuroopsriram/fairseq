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

    'augment-audio': True,
    "augmentations": "additive,pitch,speed,reverb",
    'augment-source-prob': 0.5,
    'augment-target-prob': 0.5,
    "normalize": True,
}


def w2v_base_2x150k(args, params):
    args.name = args.name or 'w2v.base.150k.augment'
    args.nodes = 2
    params['total-num-update'] = 150000
    params['max-update'] = 150000
    return args, params


def w2v_base_2x100k(args, params):
    args.name = args.name or 'w2v.base.2x100k.augment'
    args.nodes = 2
    params['total-num-update'] = 100000
    params['max-update'] = 100000
    return args, params


def w2v_base_4x250k(args, params):
    args.name = args.name or 'w2v.base.4x250k.augment'
    args.nodes = 4
    params['total-num-update'] = 250000
    params['max-update'] = 250000
    return args, params


def w2v_base_400k(args, params):
    args.name = args.name or 'w2v.base.400K.augment'
    args.nodes = 8
    params['total-num-update'] = 400000
    params['max-update'] = 400000
    return args, params


#### Sweeps

@submit.register_sweep
def sweep_w2v_base_noaug(base_args):
    lrs = [5e-4]
    # lrs = [5e-3]
    dos = [0]
    lds = [0.05]
    param_sweeps = [
        (
            f"lr{lr}.ld{ld}.noaug",
            {
                'augment-audio': False,
                "lr": lr,
                "dropout-input": do,
                "dropout-features": do,
                "dropout": do,
                "encoder-layerdrop": ld,
            },
        )
        for do in dos
        for ld in lds
        for lr in lrs
    ]
    submit.run_sweeps(w2v_base_2x100k, base_args, base_params, param_sweeps)
    # submit.run_sweeps(w2v_base_4x250k, base_args, base_params, param_sweeps)


# @submit.register_sweep
# def sweep_w2v_base3(base_args):
#     lrs = [5e-4]
#     aug_params = [
#         # (aug?, source_prob, target_prob, speed_std)
#         # (False, 0, 0, 0),
#         # (True, 0.1, 0, 0),
#         # (True, 0.1, 0, 0.001),
#         # (True, 0.1, 0, 0.01),
#
#         (True, 0, 0, 0),
#     ]
#     augmentations = ["speed"]
#     dos = [0.]
#     lds = [0.]
#     # fgms = [0.05]
#     param_sweeps = [
#         (
#             # f"lr{lr}.sourceaug{saug}.targetaug{taug}.augs{augset}.ld{ld}.spdstd{speedstd}.fgm{fgm}",
#             f"lr{lr}.sourceaug{saug}.targetaug{taug}.augs{augset}.ld{ld}.spdstd{speedstd}.noaug",
#             {
#                 "lr": lr,
#                 "dropout-input": do,
#                 "dropout-features": do,
#                 "dropout": do,
#                 "encoder-layerdrop": ld,
#
#                 "augment-audio": aug,
#                 "augmentations": augset,
#                 "augment-source-prob": saug,
#                 "augment-target-prob": taug,
#
#                 "speed-std": speedstd,
#                 # "feature-grad-mult": fgm
#                 # "normalize": False,
#             },
#         )
#         for augset in augmentations
#         for aug, saug, taug, speedstd in aug_params
#         for do in dos
#         for ld in lds
#         for lr in lrs
#         # for fgm in fgms
#     ]
#     submit.run_sweeps(w2v_base_2x100k, base_args, base_params, param_sweeps)


# @submit.register_sweep
# def sweep_w2v_base2(base_args):
#     lrs = [5e-4]
#     aug_probs = [
#         (True, 0., 0.5),  # Target augmentation
#     ]
#     speed_stds = [
#         0.01,
#         0.02,
#         0.04,
#         0.08,
#     ]
#     augmentations = [
#         "speed",
#     ]
#     dos = [0.]
#     lds = [0.]
#     param_sweeps = [
#         (
#             f"lr{lr}.sourceaug{saug}.targetaug{taug}.augs{augset}.ld{ld}.spdstd{speedstd}",
#             {
#                 "lr": lr,
#                 "dropout-input": do,
#                 "dropout-features": do,
#                 "dropout": do,
#                 "encoder-layerdrop": ld,
#
#                 "augment-audio": aug,
#                 "augmentations": augset,
#                 "augment-source-prob": saug,
#                 "augment-target-prob": taug,
#
#                 "speed-std": speedstd,
#             },
#         )
#         for augset in augmentations
#         for speedstd in speed_stds
#         for aug, saug, taug in aug_probs
#         for do in dos
#         for ld in lds
#         for lr in lrs
#     ]
#     submit.run_sweeps(w2v_base_4x250k, base_args, base_params, param_sweeps)

@submit.register_sweep
def sweep_w2v_base_aug4(base_args):
    lrs = [5e-4]
    aug_probs = [
        (True, 0, 0),  # Both augmentation
    ]
    augmentations = [
        "speed",
    ]
    dos = [0]
    lds = [0.05]
    param_sweeps = [
        (
            f"lr{lr}.sourceaug{saug}.targetaug{taug}.augs{augset}.ld{ld}",
            {
                "lr": lr,
                "dropout-input": do,
                "dropout-features": do,
                "dropout": do,
                "encoder-layerdrop": ld,

                "augment-audio": aug,
                "augmentations": augset,
                "augment-source-prob": saug,
                "augment-target-prob": taug,
            },
        )
        for augset in augmentations
        for aug, saug, taug in aug_probs
        for do in dos
        for ld in lds
        for lr in lrs
    ]
    submit.run_sweeps(w2v_base_2x100k, base_args, base_params, param_sweeps)


@submit.register_sweep
def sweep_w2v_base_8x400K_aug(base_args):
    aug_params = [
        # (Aug, SourceAug, TargetAug, Augs, AugParams)
        (True, 1, 1, "speed", [("speed-std", 0.15)]),
        (True, 1, 1, "additive", [("snr-min", 5)]),
        (True, 1, 1, "additive,speed", [("snr-min", 8), ("speed-std", 0.1)]),
        (True, 1, 1, "additive,speed", [("snr-min", 5), ("speed-std", 0.1)]),
    ]
    lrs = [5e-4]
    dos = [0]
    lds = [0.]
    param_sweeps = [
        (
            f"lr{lr}.sourceaug{srcAug}.targetaug{tgtAug}.augs{augSet}.ld{ld}." + ".".join([f"{k.replace('-', '_')}{v}" for k, v in augParams]),
            {
                "lr": lr,
                "dropout-input": do,
                "dropout-features": do,
                "dropout": do,
                "encoder-layerdrop": ld,

                "augment-audio": aug,
                "augment-source-prob": srcAug,
                "augment-target-prob": tgtAug,
                "augmentations": augSet,
                **dict(augParams),

                "total-num-update": 400000,
                "max-update": 400000,
                "update-freq": 2,
            }
        )
        for aug, srcAug, tgtAug, augSet, augParams in aug_params
        for do in dos
        for lr in lrs
        for ld in lds
    ]

    base_args.name = base_args.name or 'w2v.base.8x400k.augment'
    base_args.nodes = 4

    submit.run_sweeps(w2v_base_4x250k, base_args, base_params, param_sweeps)


@submit.register_sweep
def sweep_w2v_base_4x250K_aug(base_args):
    aug_params = [
        # (Aug, SourceAug, TargetAug, Augs, AugParams)
        # (True, 1, 1, "speed", [("speed-std", 0.1)]),
        # (True, 1, 1, "speed", [("speed-std", 0.15)]),
        # (True, 1, 1, "additive", [("snr-min", 8)]),
        # (True, 1, 1, "additive", [("snr-min", 10)]),
        # (True, 1, 1, "additive,speed", [("snr-min", 10), ("speed-std", 0.1)]),

        (True, 1, 1, "pitch", [("pitch-shift-std", 20)]),
        (True, 1, 1, "pitch", [("pitch-shift-std", 40)]),
        (True, 1, 1, "pitch", [("pitch-shift-std", 80)]),

        (True, 0.5, 0.5, "reverb", [("reverb-strength", 50), ("reverb-damping", 50), ("reverb-room-std", 10)]),
        (True, 0.5, 0.5, "reverb", [("reverb-strength", 50), ("reverb-damping", 50), ("reverb-room-std", 20)]),
        (True, 0.5, 0.5, "reverb", [("reverb-strength", 50), ("reverb-damping", 50), ("reverb-room-std", 30)]),
    ]
    lrs = [5e-4]
    dos = [0]
    lds = [0.]
    param_sweeps = [
        (
            f"lr{lr}.sourceaug{srcAug}.targetaug{tgtAug}.augs{augSet}.ld{ld}." + ".".join([f"{k.replace('-', '_')}{v}" for k, v in augParams]),
            {
                "lr": lr,
                "dropout-input": do,
                "dropout-features": do,
                "dropout": do,
                "encoder-layerdrop": ld,

                "augment-audio": aug,
                "augment-source-prob": srcAug,
                "augment-target-prob": tgtAug,
                "augmentations": augSet,
                **dict(augParams),
            }
        )
        for aug, srcAug, tgtAug, augSet, augParams in aug_params
        for do in dos
        for lr in lrs
        for ld in lds
    ]
    # for ps in param_sweeps:
    #     print(ps)
    submit.run_sweeps(w2v_base_4x250k, base_args, base_params, param_sweeps)


@submit.register_sweep
def sweep_w2v_base_aug(base_args):
    lrs = [5e-4]
    aug_probs = [
        # (True, 1, 0.),  # Source augmentation
        # (True, 0., 1),  # Target augmentation
        (True, 1, 1),  # Both augmentation
    ]
    augmentations = [
        # "additive",
        "speed",
        # "pitch",
        # "reverb",
        # "additive,pitch,speed,reverb",
        # "additive,speed"
    ]
    dos = [0]
    lds = [
        # 0.05,
        0.
    ]
    # aug_params = [
    #     # speed_std, snr_min
    #     (0.05, 5),
    #     (0.05, 7),
    #     (0.05, 10),
    #     (0.1, 5),
    #     (0.1, 7),
    #     (0.1, 10),
    #     (0.2, 5),
    #     (0.2, 7),
    #     (0.2, 10),
    # ]
    # aug_params = [
    #     # pitch_std
    #     10,
    #     20,
    #     40,
    # ]
    # aug_params = [
    #     # snr_min
    #     5,
    #     8,
    #     12,
    # ]
    aug_params = [
        # speed_std
        # 0.05,
        # 0.075,
        # 0.15,
        0.2,
        # 0.25,
        # 0.3,
    ]
    param_sweeps = [
        (
            # f"lr{lr}.sourceaug{saug}.targetaug{taug}.augs{augset}.ld{ld}.spdstd{speed_std}.snrmin{snr_min}",
            # f"lr{lr}.sourceaug{saug}.targetaug{taug}.augs{augset}.ld{ld}.pitchstd{pitchstd}",
            # f"lr{lr}.sourceaug{saug}.targetaug{taug}.augs{augset}.ld{ld}.snrmin{snr_min}",
            f"lr{lr}.sourceaug{saug}.targetaug{taug}.augs{augset}.ld{ld}.spdstd{speed_std}",
            {
                "lr": lr,
                "dropout-input": do,
                "dropout-features": do,
                "dropout": do,
                "encoder-layerdrop": ld,

                "augment-audio": aug,
                "augmentations": augset,
                "augment-source-prob": saug,
                "augment-target-prob": taug,

                "speed-std": speed_std,
                # "snr-min": snr_min,
                # "pitch-shift-std": pitchstd
            },
        )
        for augset in augmentations
        for aug, saug, taug in aug_probs
        for do in dos
        for ld in lds
        for lr in lrs
        # for speed_std, snr_min in aug_params
        # for pitchstd in aug_params
        # for snr_min in aug_params
        for speed_std in aug_params
    ]
    submit.run_sweeps(w2v_base_2x100k, base_args, base_params, param_sweeps)


@submit.register_sweep
def sweep_w2v_base_noquant(base_args):
    lrs = [5e-4]
    aug_probs = [
        (True, 1, 0.),  # Source augmentation
        (True, 0., 1),  # Target augmentation
        (True, 1, 1),  # Both augmentation
    ]
    augmentations = [
        "additive",
        "speed",
        # "pitch",
        "reverb",
        "additive,speed,reverb",
        # "additive,pitch,speed,reverb",
    ]
    dos = [0]
    lds = [0.]
    param_sweeps = [
        (
            f"noquant.lr{lr}.sourceaug{saug}.targetaug{taug}.augs{augset}.ld{ld}",
            {
                "lr": lr,
                "dropout-input": do,
                "dropout-features": do,
                "dropout": do,
                "encoder-layerdrop": ld,

                "augment-audio": aug,
                "augmentations": augset,
                "augment-source-prob": saug,
                "augment-target-prob": taug,

                "quantize-targets": False,
                "loss-weights": [10],
                "target-glu": True
            },
        )
        for augset in augmentations
        for aug, saug, taug in aug_probs
        for do in dos
        for ld in lds
        for lr in lrs
    ]
    # param_sweeps = param_sweeps[:1]
    submit.run_sweeps(w2v_base_2x100k, base_args, base_params, param_sweeps)


# @submit.register_sweep
# def sweep_w2v_base(base_args):
#     # lrs = [5e-5, 1e-3]
#     lrs = [5e-4]
#     aug_probs = [
#         # (False, 0., 0.),  # No augmentation
#         # (True, 1., 0.),  # Source augmentation
#         # (True, 0., 1.),  # Target augmentation
#         # (True, 1., 1.),  # Both augmentation
#         # (True, 0.5, 0.),  # Source augmentation
#         # (True, 0., 0.5),  # Target augmentation
#         # (True, 0.5, 0.5),  # Both augmentation
#     ]
#     param_sweeps = [
#         (
#             f'lr{lr}.sourceaug{saug}.targetaug{taug}',
#             {
#                 'lr': lr,
#                 'augment-audio': aug,
#                 'augment-source-prob': saug,
#                 'augment-target-prob': taug,
#             },
#         )
#         for lr in lrs
#         for aug, saug, taug in aug_probs
#     ]
#     submit.run_sweeps(w2v_base_2x150k, base_args, base_params, param_sweeps)
#
#
# @submit.register_sweep
# def sweep_w2v_base2(base_args):
#     lrs = [5e-4]
#     aug_probs = [
#         # (False, 0., 0.),  # No augmentation
#         # (True, 1., 0.),  # Source augmentation
#         # (True, 0., 1.),  # Target augmentation
#         # (True, 1., 1.),  # Both augmentation
#         # (True, 0.5, 0.),  # Source augmentation
#         # (True, 0., 0.5),  # Target augmentation
#         (True, 0.5, 0.5),  # Both augmentation
#     ]
#     pitch_shifts = [
#         200, 300, 400,
#     ]
#     time_drops = [
#         0.04, 0.08,
#     ]
#     param_sweeps = [
#         (
#             f'lr{lr}.sourceaug{saug}.targetaug{taug}.pshft{pshft}.tdrop{tdrop}',
#             {
#                 'lr': lr,
#                 'augment-audio': aug,
#                 'augment-source-prob': saug,
#                 'augment-target-prob': taug,
#                 'augment-pitch-shift': pshft,
#                 'augment-time-drop': tdrop,
#             },
#         )
#         for lr in lrs
#         for pshft in pitch_shifts
#         for tdrop in time_drops
#         for aug, saug, taug in aug_probs
#     ]
#     submit.run_sweeps(w2v_base_2x100k, base_args, base_params, param_sweeps)
#
#
# @submit.register_sweep
# def sweep_w2v_base3(base_args):
#     # lrs = [2e-3, 5e-4, 1e-4]
#     lrs = [5e-4]
#     aug_probs = [
#         # (False, 0., 0.),  # No augmentation
#         # (True, 1., 0.),  # Source augmentation
#         # (True, 0., 1.),  # Target augmentation
#         # (True, 1., 1.),  # Both augmentation
#         (True, 0.5, 0.),  # Source augmentation
#         (True, 0., 0.5),  # Target augmentation
#         (True, 0.5, 0.5),  # Both augmentation
#     ]
#     # mask_probs = [0.325, 0.65]
#     # dropouts = [0.05, 0.1]
#     mask_probs = [0.1, 0.2]
#     dropouts = [0.01, 0.05]
#
#     param_sweeps = [
#         (
#             f'lr{lr}.sourceaug{saug}.targetaug{taug}.mp{mp}.do{do}.additive',
#             {
#                 'lr': lr,
#                 'augment-audio': aug,
#                 'augment-source-prob': saug,
#                 'augment-target-prob': taug,
#
#                 'mask-prob': mp,
#
#                 'dropout-input': do,
#                 'dropout-features': do,
#                 'dropout': do,
#                 'attention-dropout': do,
#             },
#         )
#         for lr in lrs
#         for aug, saug, taug in aug_probs
#         for mp in mask_probs
#         for do in dropouts
#     ]
#     submit.run_sweeps(w2v_base_2x100k, base_args, base_params, param_sweeps)
#
#
# @submit.register_sweep
# def sweep_w2v_base4(base_args):
#     lrs = [5e-4]
#     aug_probs = [
#         (False, 0., 0.),  # No augmentation
#     ]
#     param_sweeps = [
#         (
#             f'lr{lr}.sourceaug{saug}.targetaug{taug}',
#             {
#                 'lr': lr,
#                 'augment-audio': aug,
#                 'augment-source-prob': saug,
#                 'augment-target-prob': taug,
#             },
#         )
#         for lr in lrs
#         for aug, saug, taug in aug_probs
#     ]
#     submit.run_sweeps(w2v_base_2x100k, base_args, base_params, param_sweeps)
#
#
# @submit.register_sweep
# def sweep_w2v_base3_tgtglu(base_args):
#     # lrs = [2e-3, 5e-4, 1e-4]
#     lrs = [5e-4]
#     aug_probs = [
#         # (False, 0., 0.),  # No augmentation
#         # (True, 1., 0.),  # Source augmentation
#         # (True, 0., 1.),  # Target augmentation
#         # (True, 1., 1.),  # Both augmentation
#         (True, 0.5, 0.),  # Source augmentation
#         (True, 0., 0.5),  # Target augmentation
#         (True, 0.5, 0.5),  # Both augmentation
#     ]
#     mask_probs = [0.325, 0.65]
#     dropouts = [0.05, 0.1]
#     # mask_probs = [0.1, 0.2]
#     # dropouts = [0.01, 0.05]
#
#     param_sweeps = [
#         (
#             f'lr{lr}.sourceaug{saug}.targetaug{taug}.mp{mp}.do{do}.additive.tgtglu',
#             {
#                 'lr': lr,
#                 'augment-audio': aug,
#                 'augment-source-prob': saug,
#                 'augment-target-prob': taug,
#
#                 'mask-prob': mp,
#
#                 'dropout-input': do,
#                 'dropout-features': do,
#                 'dropout': do,
#                 'attention-dropout': do,
#
#                 "target-glu": True
#             },
#         )
#         for lr in lrs
#         for aug, saug, taug in aug_probs
#         for mp in mask_probs
#         for do in dropouts
#     ]
#     submit.run_sweeps(w2v_base_2x100k, base_args, base_params, param_sweeps)
#
#
# @submit.register_sweep
# def sweep_w2v_base3_noquant(base_args):
#     # lrs = [2e-3, 5e-4, 1e-4]
#     lrs = [5e-4]
#     aug_probs = [
#         # (False, 0., 0.),  # No augmentation
#         # (True, 1., 0.),  # Source augmentation
#         # (True, 0., 1.),  # Target augmentation
#         # (True, 1., 1.),  # Both augmentation
#         (True, 0.5, 0.),  # Source augmentation
#         (True, 0., 0.5),  # Target augmentation
#         (True, 0.5, 0.5),  # Both augmentation
#     ]
#     mask_probs = [0.325, 0.65]
#     dropouts = [0.05, 0.1]
#     # mask_probs = [0.1, 0.2]
#     # dropouts = [0.01, 0.05]
#
#     param_sweeps = [
#         (
#             f'lr{lr}.sourceaug{saug}.targetaug{taug}.mp{mp}.do{do}.additive.noquant',
#             {
#                 'lr': lr,
#                 'augment-audio': aug,
#                 'augment-source-prob': saug,
#                 'augment-target-prob': taug,
#
#                 'mask-prob': mp,
#
#                 'dropout-input': do,
#                 'dropout-features': do,
#                 'dropout': do,
#                 'attention-dropout': do,
#
#                 "quantize-targets": False,
#                 "loss-weights": [10]
#             },
#         )
#         for lr in lrs
#         for aug, saug, taug in aug_probs
#         for mp in mask_probs
#         for do in dropouts
#     ]
#     submit.run_sweeps(w2v_base_2x100k, base_args, base_params, param_sweeps)


if __name__ == '__main__':
    parser = submit.create_parser()
    base_args = parser.parse_args()
    sweep_func = submit.get_sweep_func(base_args.sweep)
    sweep_func(base_args)
