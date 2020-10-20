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


def w2v_base_250k(args, params):
    args.name = args.name or 'w2v.base.250k'
    args.nodes = 4
    params['total-num-update'] = 2500000
    params['max-update'] = 2500000
    return args, params


def w2v_base_400k(args, params):
    args.name = args.name or 'w2v.base.400K'
    args.nodes = 8
    params['total-num-update'] = 400000
    params['max-update'] = 400000
    return args, params


def w2v_large_600k(args, params):
    # 317M params
    args.name = args.name or 'w2v.large.600K'
    args.nodes = 16
    params.update({
        'final-dim': 768,
        'latent-temp': (2.0, 0.1, 0.999995),
        'total-num-update': 600000,
        'encoder-layerdrop': 0.,
        'feature-grad-mult': 0.03,
        'encoder-layers': 24,
        'encoder-embed-dim': 1024,
        'encoder-ffn-embed-dim': 4096,
        'encoder-attention-heads': 16,
        'max-sample-size': 320000,
        'dropout': 0.0,
        'max-tokens': 1200000,
        'max-update': 600000,
    })
    return args, params


def w2v_conformer_250k(args, params):
    args.name = args.name or 'w2v.conformer.250k'
    args.nodes = 4
    params.update({
        'transformer-type': 'conformer',
        'encoder-layers': 17,
        'encoder-embed-dim': 512,
        'encoder-ffn-embed-dim': 512,
        'encoder-attention-heads': 8,
        'total-num-update': 250000,
        'max-update': 250000,
    })
    return args, params


def w2v_conformer_400k(args, params):
    args.name = args.name or 'w2v.conformer.400k'
    args.nodes = 8
    params.update({
        'transformer-type': 'conformer',
        'encoder-layers': 17,
        'encoder-embed-dim': 512,
        'encoder-ffn-embed-dim': 512,
        'encoder-attention-heads': 8,
        'total-num-update': 400000,
        'max-update': 400000,
    })
    return args, params


def w2v_conformer_relpos_250k(args, params):
    args.name = args.name or 'w2v.conformer.relpos.250k'
    args.nodes = 4
    params.update({
        'transformer-type': 'conformer',
        'encoder-layers': 17,
        'encoder-embed-dim': 512,
        'encoder-ffn-embed-dim': 512,
        'encoder-attention-heads': 8,
        'total-num-update': 250000,
        'max-update': 250000,
        'use-rel-posn-mha': True,
        'num-relpos-embeds': 16,
        'lin-dropout': 0.1,
    })
    return args, params


def w2v_conformer_relpos_2x100k(args, params):
    args.name = args.name or 'w2v.conformer.relpos.2x100k'
    args.nodes = 2
    params.update({
        'transformer-type': 'conformer',
        'encoder-layers': 17,
        'encoder-embed-dim': 512,
        'encoder-ffn-embed-dim': 512,
        'encoder-attention-heads': 8,
        'total-num-update': 100000,
        'max-update': 100000,
        'use-rel-posn-mha': True,
        'num-relpos-embeds': 16,
        'lin-dropout': 0.1,
    })
    return args, params


def w2v_conformer_relpos_400k(args, params):
    args.name = args.name or 'w2v.conformer.relpos.400k'
    args.nodes = 8
    params.update({
        'transformer-type': 'conformer',
        'encoder-layers': 17,
        'encoder-embed-dim': 512,
        'encoder-ffn-embed-dim': 512,
        'encoder-attention-heads': 8,
        'total-num-update': 400000,
        'max-update': 400000,
        'use-rel-posn-mha': True,
        'num-relpos-embeds': 16,
        'lin-dropout': 0.1,
    })
    return args, params


def w2v_conformer_relpos_large_21lyrs_600k(args, params):
    args.name = args.name or 'w2v.conformer.relpos.600K.16nd'
    args.nodes = 16

    params.update({
        'final-dim': 768,  # Less than encoder-embed-dim??
        'latent-temp': (2.0, 0.1, 0.999995),
        'total-num-update': 600000,
        'encoder-layerdrop': 0.,
        'feature-grad-mult': 0.03,
        'encoder-layers': 21,
        'encoder-embed-dim': 768,
        'max-sample-size': 320000,
        'dropout': 0.0,
        'max-tokens': 1200000,
        'max-update': 600000,

        'transformer-type': 'conformer',
        'encoder-attention-heads': 8,

        'use-rel-posn-mha': True,
        'num-relpos-embeds': 16,
        'lin-dropout': 0.,
    })
    return args, params


def w2v_conformer_relpos_large_32lyrs_600k(args, params):
    args.name = args.name or 'w2v.conformer.relpos.large.32lyrs.600K.16nd'
    args.nodes = 16

    params.update({
        'final-dim': 768,
        'latent-temp': (2.0, 0.1, 0.999995),
        'total-num-update': 600000,
        'encoder-layerdrop': 0.,
        'feature-grad-mult': 0.03,
        'encoder-layers': 32,
        'encoder-embed-dim': 640,
        'max-sample-size': 320000,
        'dropout': 0.0,
        'max-tokens': 1200000,
        'max-update': 600000,

        'transformer-type': 'conformer',
        'encoder-attention-heads': 8,

        'use-rel-posn-mha': True,
        'num-relpos-embeds': 16,
        'lin-dropout': 0.,
    })
    return args, params


def w2v_conformer_relpos_large_21lyrs_600k_8nodes(args, params):
    args.name = args.name or 'w2v.conformer.relpos.600K'
    args.nodes = 8

    params.update({
        'final-dim': 768,  # Less than encoder-embed-dim??
        'latent-temp': (2.0, 0.1, 0.999995),
        'total-num-update': 600000,
        'encoder-layerdrop': 0.,
        'feature-grad-mult': 0.03,
        'encoder-layers': 21,
        'encoder-embed-dim': 768,
        'max-sample-size': 320000,
        'dropout': 0.0,
        'max-tokens': 1200000,
        'max-update': 600000,

        'transformer-type': 'conformer',
        'encoder-attention-heads': 8,

        'use-rel-posn-mha': True,
        'num-relpos-embeds': 16,
        'lin-dropout': 0.,

        'update-freq': 2,
    })
    return args, params


def w2v_conformer_relpos_large_26lyrs_600k_8nodes(args, params):
    args.name = args.name or 'w2v.conformer.relpos.600K'
    args.nodes = 8

    params.update({
        'final-dim': 688,  # Less than encoder-embed-dim??
        'latent-temp': (2.0, 0.1, 0.999995),
        'total-num-update': 600000,
        'encoder-layerdrop': 0.,
        'feature-grad-mult': 0.03,
        'encoder-layers': 24,
        'encoder-embed-dim': 688,
        'max-sample-size': 320000,
        'dropout': 0.0,
        'max-tokens': 1200000,
        'max-update': 600000,

        'transformer-type': 'conformer',
        'encoder-attention-heads': 8,

        'use-rel-posn-mha': True,
        'num-relpos-embeds': 16,
        'lin-dropout': 0.,

        'update-freq': 2,
    })
    return args, params


#### Sweeps

@submit.register_sweep
def sweep_w2v_base_250k_17lyrs(base_args):
    # 110M params
    dims = [704]
    # lrs = [5e-4]
    lrs = [1e-4, 2e-3]
    encoder_layers = [17]
    param_sweeps = [
        (
            f'dim{dim}.enclyrs{enc_lyrs}.lr{lr}',
            {
                'encoder-embed-dim': dim,
                'encoder-ffn-embed-dim': dim * 4,
                'lr': lr,
                'encoder-layers': enc_lyrs,
                'encoder-attention-heads': 8,
            },
        )
        for dim in dims
        for lr in lrs
        for enc_lyrs in encoder_layers
    ]
    submit.run_sweeps(w2v_base_250k, base_args, base_params, param_sweeps)


@submit.register_sweep
def sweep_w2v_large_600k(base_args):
    lrs = [3e-4]
    param_sweeps = [
        (
            f'lr{lr}',
            {
                'lr': lr,
            },
        )
        for lr in lrs
    ]
    submit.run_sweeps(w2v_large_600k, base_args, base_params, param_sweeps)


@submit.register_sweep
def sweep_w2v_conformer_250k_12lyrs(base_args):
    # 104M params
    dims = [576]
    lrs = [5e-4, 1e-3]
    encoder_layers = [12]
    param_sweeps = [
        (
            f'dim{dim}.enclyrs{enc_lyrs}.lr{lr}',
            {
                'encoder-embed-dim': dim,
                'encoder-ffn-embed-dim': dim,
                'lr': lr,
                'encoder-layers': enc_lyrs,
            },
        )
        for dim in dims
        for lr in lrs
        for enc_lyrs in encoder_layers
    ]
    submit.run_sweeps(w2v_conformer_250k, base_args, base_params, param_sweeps)


@submit.register_sweep
def sweep_w2v_conformer_400k_17lyrs(base_args):
    dims = [512]
    # lrs = [5e-4, 1e-3]
    lrs = [5e-4]
    encoder_layers = [17]
    param_sweeps = [
        (
            f'dim{dim}.enclyrs{enc_lyrs}.lr{lr}',
            {
                'encoder-embed-dim': dim,
                'encoder-ffn-embed-dim': dim,
                'lr': lr,
                'encoder-layers': enc_lyrs,
            },
        )
        for dim in dims
        for lr in lrs
        for enc_lyrs in encoder_layers
    ]
    submit.run_sweeps(w2v_conformer_400k, base_args, base_params, param_sweeps)


@submit.register_sweep
def sweep_w2v_conformer_relpos_250k_17lyrs(base_args):
    dims = [512]
    num_relpos_embeds = [16]
    lrs = [2e-4, 5e-4, 1e-3]
    encoder_layers = [17]
    param_sweeps = [
        (
            f'dim{dim}.enclyrs{enc_lyrs}.lr{lr}.rpemb{rpemb}',
            {
                'encoder-embed-dim': dim,
                'encoder-ffn-embed-dim': dim,
                'lr': lr,
                'encoder-layers': enc_lyrs,
                'num-relpos-embeds': rpemb,
            },
        )
        for dim in dims
        for lr in lrs
        for enc_lyrs in encoder_layers
        for rpemb in num_relpos_embeds
    ]
    submit.run_sweeps(w2v_conformer_relpos_250k, base_args, base_params, param_sweeps)


@submit.register_sweep
def sweep_w2v_conformer_partial_relpos_250k_17lyrs(base_args):
    dims = [512]
    num_relpos_embeds = [16]
    lrs = [1e-3]
    encoder_layers = [17]
    rel_posn_mha_list = {
        # 'none': (False,) * 8
        # 'all': (True,) * 8,

        'first4': (True,) * 4 + (False,) * 13,
        'last4': (False,) * 13 + (True,) * 4,

        # 'first8': (True,) * 8 + (False,) * 9,
        # 'last8': (False,) * 9 + (True,) * 8,
        # 'alt': (False, True) * 8 + (False,),
    }
    param_sweeps = [
        (
            f'dim{dim}.enclyrs{enc_lyrs}.lr{lr}.rpemb{rpemb}.mha{name}',
            {
                'encoder-embed-dim': dim,
                'encoder-ffn-embed-dim': dim,
                'lr': lr,
                'encoder-layers': enc_lyrs,
                'num-relpos-embeds': rpemb,
                'rel-posn-mha-list': mhalst,
            },
        )
        for dim in dims
        for lr in lrs
        for enc_lyrs in encoder_layers
        for rpemb in num_relpos_embeds
        for name, mhalst in rel_posn_mha_list.items()
    ]
    submit.run_sweeps(w2v_conformer_relpos_250k, base_args, base_params, param_sweeps)


@submit.register_sweep
def sweep_w2v_conformer_relpos_400k_17lyrs(base_args):
    dims = [512]
    num_relpos_embeds = [16]
    lrs = [2e-4, 5e-4]
    encoder_layers = [17]
    param_sweeps = [
        (
            f'dim{dim}.enclyrs{enc_lyrs}.lr{lr}.rpemb{rpemb}',
            {
                'encoder-embed-dim': dim,
                'encoder-ffn-embed-dim': dim,
                'lr': lr,
                'encoder-layers': enc_lyrs,
                'num-relpos-embeds': rpemb,
            },
        )
        for dim in dims
        for lr in lrs
        for enc_lyrs in encoder_layers
        for rpemb in num_relpos_embeds
    ]
    submit.run_sweeps(w2v_conformer_relpos_400k, base_args, base_params, param_sweeps)


@submit.register_sweep
def sweep_w2v_conformer_relpos_large_32lyrs_600k(base_args):
    lrs = [8e-4]
    param_sweeps = [
        (
            f'lr{lr}',
            {
                'lr': lr,
                'min-loss-scale': 0.001,

                'encoder-layerdrop': 0.075,
                'dropout': 0.15,
                'attention-dropout': 0.15,
                'dropout-features': 0.15,
                'dropout-input': 0.15,
                'lin-dropout': 0.15,
            },
        )
        for lr in lrs
    ]
    submit.run_sweeps(w2v_conformer_relpos_large_32lyrs_600k, base_args, base_params, param_sweeps)



@submit.register_sweep
def sweep_w2v_conformer_relpos_large_21lyrs_600k(base_args):
    # lrs = [1e-4, 3e-4]
    lrs = [1e-3]
    param_sweeps = [
        (
            f'lr{lr}',
            {
                'lr': lr / 4,
                'end-learning-rate': lr / 16,
                'min-loss-scale': 0.01,

                'encoder-layerdrop': 0.075,
                'dropout': 0.15,
                'lin-dropout': 0.15,
                'attention-dropout': 0.15,
            },
        )
        for lr in lrs
    ]
    submit.run_sweeps(w2v_conformer_relpos_large_21lyrs_600k, base_args, base_params, param_sweeps)


@submit.register_sweep
def sweep_w2v_conformer_relpos_large_21lyrs_600k_librivox(base_args):
    base_args.name = 'w2v.conformer.relpos.600K.16nd.librivox'
    lrs = [1e-3]
    param_sweeps = [
        (
            f'lr{lr}',
            {
                'lr': lr / 4,
                'end-learning-rate': lr / 8,
                'min-loss-scale': 0.05,

                'encoder-layerdrop': 0.01,
                'dropout': 0.05,
                'lin-dropout': 0.05,
                'attention-dropout': 0.1,
            },
        )
        for lr in lrs
    ]
    submit.run_sweeps(w2v_conformer_relpos_large_21lyrs_600k, base_args, base_params, param_sweeps, dataset='librivox')


@submit.register_sweep
def sweep_w2v_conformer_relpos_large_21lyrs_600k_8nodes(base_args):
    # lrs = [1e-4, 3e-4]
    lrs = [3e-4]
    param_sweeps = [
        (
            f'lr{lr}',
            {
                'lr': lr,
            },
        )
        for lr in lrs
    ]
    submit.run_sweeps(w2v_conformer_relpos_large_21lyrs_600k_8nodes, base_args, base_params, param_sweeps)


@submit.register_sweep
def sweep_w2v_conformer_relpos_large_26lyrs_600k_8nodes(base_args):
    lrs = [1e-4, 3e-4]
    param_sweeps = [
        (
            f'lr{lr}',
            {
                'lr': lr,
            },
        )
        for lr in lrs
    ]
    submit.run_sweeps(w2v_conformer_relpos_large_26lyrs_600k_8nodes, base_args, base_params, param_sweeps)


if __name__ == '__main__':
    parser = submit.create_parser()
    base_args = parser.parse_args()
    sweep_func = submit.get_sweep_func(base_args.sweep)
    sweep_func(base_args)
