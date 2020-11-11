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
    args.name = args.name or 'w2v.base.100k.augment'
    args.nodes = 2
    params['total-num-update'] = 100000
    params['max-update'] = 100000
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
    # lrs = [5e-4]
    lrs = [5e-3]
    dos = [0]
    lds = [0.05]
    param_sweeps = [
        (
            f"lr{lr}.noaug",
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


@submit.register_sweep
def sweep_w2v_base(base_args):
    # lrs = [5e-4]
    lrs = [1e-3, 1e-5]
    # lrs = [5e-3]
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
        # "additive,pitch,speed,reverb",
    ]
    dos = [0]
    # lds = [0.05]
    lds = [0.]
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
