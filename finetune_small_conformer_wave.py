from copy import deepcopy
from pathlib import Path

import submit

base_params = {
    'distributed-world-size': 24,
    'distributed-port':13434,
    'save-dir': '/checkpoint/anuroops/fairseq/wav2vec/w2v.small.ft/',
    'fp16': True,
    # 'wer-args': ('/datasets01_101/librispeech/021419/lm/4-gram.arpa', '/checkpoint/anuroops/data/libris/lab/dict.ltr.txt', 2, -1),
    'post-process': 'letter',
    'valid-subset': 'dev_other',
    'no-epoch-checkpoints': True,
    'best-checkpoint-metric': 'wer',
    'num-workers': 4,
    'max-update': 80000,
    'sentence-avg': True,
    'task': 'audio_pretraining',
    'arch': 'wav2vec_ctc',
    'w2v-path': '',
    'labels': 'ltr',
    'apply-mask': True,
    'mask-selection': 'static',
    'mask-other': 0,
    'mask-length': 10,
    'mask-prob': 0.5,
    'layerdrop': 0.1,
    'mask-channel-selection': 'static',
    'mask-channel-other': 0,
    'mask-channel-length': 64,
    'mask-channel-prob': 0.5,
    'zero-infinity': True,
    'feature-grad-mult': 0.0,
    'freeze-finetune-updates': 10000,
    'validate-after-updates': 10000,
    'optimizer': 'adam',
    'adam-betas': (0.9, 0.98),
    'adam-eps': 1e-08,
    'lr': 2e-05,
    'lr-scheduler': 'tri_stage',
    'warmup-steps': 8000,
    'hold-steps': 32000,
    'decay-steps': 40000,
    'final-lr-scale': 0.05,
    'final-dropout': 0.0,
    'dropout': 0.0,
    'activation-dropout': 0.1,
    'criterion': 'ctc',
    'attention-dropout': 0.0,
    'max-tokens': 1280000,
    'seed': 2337,
    'log-format': 'json',
    'log-interval': 500,
    'ddp-backend': 'no_c10d',

    'no-expand-ffn': True,  # TODO
}


if __name__ == '__main__':
    parser = submit.create_parser()
    base_args = parser.parse_args()
    # if args.nodes != 3:
    #     base_params['update-freq'] = 24 / args.nodes / 8
    lrs = [6e-06, 2e-05, 6e-05]
    checkpoints = [   # ('w2v-path', 'no-expand-ffn')
        # ('logs/w2v.small.conformer.wave/dim512.lr0.0005/checkpoint_best.pt', True),
        # ('logs/w2v.small.conformer.wave2/dim512.lr0.001/checkpoint_best.pt', False),
        ('logs/w2v.small.conformer.wave2/dim512.lr0.0005/checkpoint_best.pt', False),
    ]
    param_sweeps = [
        (
            f'lr{lr}.expand{not no_expand_ffn}',
            {
                'w2v-path': w2v_path,
                'no-expand-ffn': no_expand_ffn,
                'lr': lr,
            },
        )
        for w2v_path, no_expand_ffn in checkpoints
        for lr in lrs
    ]
    for name, overrides in param_sweeps:
        args = deepcopy(base_args)
        w2v_model = Path(overrides['w2v-path']).parent.name
        args.name = f'{base_args.name}/{w2v_model}/{name}'
        params = deepcopy(base_params)
        params.update(**overrides)
        print(args.name, overrides)
        submit.main(args, params, 'lab.10h')
