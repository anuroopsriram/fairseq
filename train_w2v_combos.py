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
def w2v_base_conf_mlp(base_args):
    lrs = [
        5e-4,
    ]
    num_layers = 14
    trans_types = {
        # "conf": ",".join(["conf" for _ in range(num_layers)]),
        "conf_rp": ",".join(["conf_relpos" for _ in range(num_layers)]),
    }
    norms = [
        "batchnorm",
        # "layernorm",
    ]
    kern_sizes = [3]
    mlp_params = {
        # (mlpContext, mlpTarget, BatchNorm, Scale, Activation, Num Hidden)
        "mlpBoth": (True, True, True, 4, "relu", 2),
    }
    run_args_list = {
        "ls960h": dict(name="combo.conf2_mlp.ls960h.3x400", updates=400_000, nodes=3, update_freq=1),
    }
    for dset, run_args in run_args_list.items():
        param_sweeps = [
            (
                f"{dset}.{name}_mlp.lr{lr}.ks{kern_sz}.norm{norm}.mlp{nameMlp}.nhid{nhidden}",
                {
                    "lr": lr,
                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": run_args["update_freq"],

                    "encoder-layers": num_layers,
                    "transformer-type": trans_type,
                    "conformer-kernel-size": kern_sz,
                    'encoder-embed-dim': 512,
                    'encoder-attention-heads': 8,
                    "conformer-norm": norm,

                    "projection-mlp-context": contextMLP,
                    "target-mlp-context": targetMLP,
                    "mlp-batch-norm": batchnorm,
                    "mlp-scale": scale,
                    "mlp-activation": activation,
                    "mlp-nhidden": nhidden,

                    "activation-fn2": "glu",
                },
            )
            for lr in lrs
            for name, trans_type in trans_types.items()
            for kern_sz in kern_sizes
            for norm in norms

            for nameMlp, (contextMLP, targetMLP, batchnorm, scale, activation, nhidden) in mlp_params.items()
        ]
        args = deepcopy(base_args)
        args.name = args.name or run_args["name"]
        args.nodes = run_args["nodes"]
        submit.run_sweeps(args, base_params, param_sweeps, dataset=dset)


@submit.register_sweep
def w2v_base_conf_mlp_lconv(base_args):
    lrs = [
        5e-4,
    ]
    num_layers = 14
    trans_types = {
        "conf": ",".join(["conf" for _ in range(num_layers)]),
        "conf_rp": ",".join(["conf_relpos" for _ in range(num_layers)]),
    }
    norms = {
        "conf": "batchnorm",
        "conf_rp": "layernorm",
    }
    kern_sizes = [3]
    mlp_params = {
        # (mlpContext, mlpTarget, BatchNorm, Scale, Activation, Num Hidden)
        "mlpBoth": (True, True, True, 4, "relu", 2),
    }
    conv_feature_layers2 = [(640, 10, 5)] + [(640, 3, 2)] * 4 + [(640, 3, 2)] * 2

    num_lyrs = len(conv_feature_layers2)
    conv_types = {
        "lc_last2":  (["conv"] * (num_lyrs-2) + ["light_conv"] * 2, conv_feature_layers2),
        "dc_last2":  (["conv"] * (num_lyrs-2) + ["dynamic_conv"] * 2, conv_feature_layers2),
    }
    lconv_params = [
        # (encglu, wtsmax, wtdo, reldo, inpdo, normbef, attnhds, avgpool, mod)
        (True, True, 0.1, 0.1, 0.1, False, 8, False, "do0.1"),
    ]

    run_args_list = {
        "ls960h": dict(name="combo.conf2_mlp_lconv.ls960h.3x400", updates=400_000, nodes=3, update_freq=1),
    }
    for dset, run_args in run_args_list.items():
        param_sweeps = [
            (
                f"{dset}.{name}_mlp_{nameConv}.lr{lr}.ks{kern_sz}.norm{norms[name]}.mlp{nameMlp}.nhid{nhidden}.mod{mod}",
                {
                    "lr": lr,
                    "total-num-update": run_args["updates"],
                    "max-update": run_args["updates"],
                    "update-freq": run_args["update_freq"],

                    # Conformer
                    "encoder-layers": num_layers,
                    "transformer-type": trans_type,
                    "conformer-kernel-size": kern_sz,
                    'encoder-embed-dim': 512,
                    'encoder-attention-heads': 8,
                    "conformer-norm": norms[name],

                    # MLP
                    "projection-mlp-context": contextMLP,
                    "target-mlp-context": targetMLP,
                    "mlp-batch-norm": batchnorm,
                    "mlp-scale": scale,
                    "mlp-activation": activation,
                    "mlp-nhidden": nhidden,

                    # lconv
                    "conv-feature-layers": conv_feat_lyrs,
                    "conv-types": ",".join(ctypes),
                    "lconv-encoder-noglu": encglu,
                    "lconv-weight-nosoftmax": wtsmax,
                    "lconv-weight-dropout": wtdo,
                    "lconv-relu-dropout": reldo,
                    "lconv-input-dropout": inpdo,
                    "lconv-encoder-normalize-before": normbef,
                    "lconv-encoder-attention-heads": attnhds,
                    "lconv-avg-pool": avgpool,

                    "activation-fn2": "glu",
                },
            )
            for lr in lrs
            for name, trans_type in trans_types.items()
            for kern_sz in kern_sizes
            # for norm in norms

            for nameMlp, (contextMLP, targetMLP, batchnorm, scale, activation, nhidden) in mlp_params.items()

            for nameConv, (ctypes, conv_feat_lyrs) in conv_types.items()
            for encglu, wtsmax, wtdo, reldo, inpdo, normbef, attnhds, avgpool, mod in lconv_params
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
