from copy import deepcopy

import submit
import numpy as np

base_params = {
    'save-dir': '',
    'fp16': True,
    'distributed-world-size': 1,
    'distributed-port': 13356,
    'num-workers': 0,
    'task': 'audio_pretraining',
    'criterion': 'wav2vec',
    'arch': 'wav2vec2',
    'log-keys': ["prob_perplexity","code_perplexity","temp"],
    'quantize-targets': True,
    'extractor-mode': 'default',
    'conv-feature-layers': [(512, 3, 2)] * 2,
    'final-dim': 256,
    'latent-vars': 320,
    'latent-groups': 2,
    'latent-temp': (2,0.5,0.999995),
    'infonce': True,
    'optimizer': 'adam',
    'adam-betas': (0.9,0.98),
    'adam-eps': 1e-09,
    'lr-scheduler': 'polynomial_decay',
    'total-num-update': 250000,
    'lr': 0.0005,
    'warmup-updates': 32000,
    'mask-length': 10,
    'mask-prob': 0.65,
    'mask-selection': 'static',
    'mask-other': 0,
    'encoder-layerdrop': 0.,
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
    'weight-decay': 1e-6,
    'max-tokens': 1400000,
    'max-update': 400000,
    'skip-invalid-size-inputs-valid-test': True,
    'ddp-backend no_c10d': True,
    'update-freq': 1,

    'transformer-type': 'conformer',
    'encoder-layers': 17,
    'encoder-embed-dim': 768,
    'encoder-ffn-embed-dim': 768,
    'encoder-attention-heads': 8,

    'logmel': True,
    'in-d': 80,
}


if __name__ == '__main__':
    parser = submit.create_parser()
    base_args = parser.parse_args()

    # if args.nodes != 4:
    #     base_params['update-freq'] = 32 / args.nodes / 8

    dims = [
        # (16, 144),
        (16, 256),
        # (17, 512)
    ]
    lrs = [5e-4]
    param_sweeps = [
        (
            f'dim{dim}.lyr{lyr}.lr{lr}',
            {
                'encoder-layers': lyr,
                'conv-feature-layers': [(dim, 3, 2)] * 2,
                'encoder-embed-dim': dim,
                'encoder-ffn-embed-dim': dim,
                'lr': lr / np.sqrt(dim),
            },
        )
        for lyr, dim in dims
        for lr in lrs
    ]
    for name, overrides in param_sweeps:
        args = deepcopy(base_args)
        args.name = f'{base_args.name}/{name}'
        params = deepcopy(base_params)
        params.update(**overrides)
        print(args.name, overrides)
        submit.main(args, params, 'lab.960h')
