from copy import deepcopy

import submit

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
    # TODO: Use Conv2D
    'conv-feature-layers': [(512, 1, 2)] + [(512, 1, 1)] * 4 + [(512, 1, 1)] * 2,
    'encoder-layers': 13,
    'final-dim': 256,
    'latent-vars': 320,
    'latent-groups': 2,
    'latent-temp': (2,0.5,0.999995),
    'infonce': True,
    'optimizer': 'adam',
    'adam-betas': (0.9,0.98),
    'adam-eps': 1e-06,
    'lr-scheduler': 'polynomial_decay',
    'total-num-update': 250000,
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
    'logmel': True,
    'in-d': 80,
}


if __name__ == '__main__':
    parser = submit.create_parser()
    base_args = parser.parse_args()
    # if base_args.nodes != 4:
    #     base_params['update-freq'] = 4 / base_args.nodes

    name = base_args.name
    # lrs = [0.0005, 0.001]
    lrs = [0.0005]
    clipnorms = [0]
    for lr in lrs:
        for clipnorm in clipnorms:
            params = deepcopy(base_params)
            args = deepcopy(base_args)
            params['lr'] = lr
            params['clip-norm'] = clipnorm
            args.name = f'{base_args.name}/lr{lr}.clipnorm{clipnorm}'
            submit.main(args, params, 'unlab')
