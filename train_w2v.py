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
def w2v_base_mlp(base_args):
    lrs = [5e-4]
    mlp_params = {
        # (mlpContext, mlpTarget, BatchNorm, Scale, Activation)
        (True, True, True, 4, "relu"),
    }
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
                
                "total-num-update": 100_000,
                "max-update": 100_000,
                "update-freq": 1,
            },
        )
        for contextMLP, targetMLP, batchnorm, scale, activation in mlp_params
        for lr in lrs
    ]
    base_args.name = "w2v.base.mlp.2x100"
    base_args.nodes = 2
    submit.run_sweeps(base_args, base_params, param_sweeps)


if __name__ == '__main__':
    parser = submit.create_parser()
    base_args = parser.parse_args()
    sweep_func = submit.get_sweep_func(base_args.sweep)
    sweep_func(base_args)
