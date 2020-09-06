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
    'w2v-path': '/checkpoint/anuroops/fairseq/wav2vec/w2v.small/checkpoint_best.pt',
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
}


if __name__ == '__main__':
    parser = submit.create_parser()
    args = parser.parse_args()
    if args.nodes != 3:
        base_params['update-freq'] = 24 / args.nodes / 8
    submit.main(args, base_params, 'lab.10h')
