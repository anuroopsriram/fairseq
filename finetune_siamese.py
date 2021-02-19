from copy import deepcopy
from pathlib import Path

import numpy as np

import submit

wer_args = (
    '/datasets01_101/librispeech/021419/lm/4-gram.arpa',
    '/checkpoint/anuroops/data/libris/lab.960h/lexicon_ltr.lst',
    2, -1,
)


base_params = {
    'distributed-world-size': 24,
    'distributed-port': 13434,
    'save-dir': '/checkpoint/anuroops/fairseq/wav2vec/w2v.base.ft/',
    'fp16': True,
    'post-process': 'letter',
    'valid-subset': 'dev_other',
    'no-epoch-checkpoints': True,
    'best-checkpoint-metric': 'wer',
    'num-workers': 4,
    'max-update': 320000,
    'sentence-avg': True,
    'task': 'audio_pretraining',
    'arch': 'wav2vec_ctc',
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
    # 'freeze-finetune-updates': 10000,
    'freeze-finetune-updates': 0,
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
    # 'log-format': 'json',
    'log-interval': 500,
    'ddp-backend': 'no_c10d',
    'validate-interval-updates': 500,
    'validate-interval': 10000,
    'save-interval-updates': 500,
    'no-epoch-checkpoints': True,
}


@submit.register_sweep
def w2v_siamese(base_args):
    checkpoints = {
        "siamese.1x100.ft": [
            "logs/siamese.1x100/lr0.0002.mp0.0speed-std0.1.unlab",
            # "logs/siamese.1x100/lr0.0002.mp0.0speed-std0.2.unlab",
            # "logs/siamese.1x100/lr0.0005.mp0.0speed-std0.2.unlab",
            # "logs/siamese.1x100/lr0.0005.mp0.25speed-std0.1.unlab",
            # "logs/siamese.1x100/lr0.0005.mp0.5speed-std0.05.unlab",
            # "logs/siamese.1x100/lr0.0005.mp0.65speed-std0.1_snr-min10_snr-max15.unlab",
            # "logs/siamese.1x100/lr0.002.mp0.0speed-std0.1.unlab",
            # "logs/siamese.1x100/lr0.002.mp0.5speed-std0.05.unlab",
            # "logs/siamese.1x100/lr0.002.mp0.5speed-std0.1.unlab",
            # "logs/siamese.1x100/lr0.002.mp0.5speed-std0.2.unlab",
            # "logs/siamese.1x100/lr0.002.mp0.65speed-std0.15_snr-min5_snr-max15_pitch-shift-std200.unlab",
            # "logs/siamese.1x100/lr0.002.mp0.65speed-std0.1_snr-min10_snr-max15.unlab",
            # "logs/siamese.1x100/lr5e-05.mp0.25speed-std0.2.unlab",
        ],
    }
    max_update = 25_000
    lrs = [2e-05]
    mask_lens = [4, 10]
    mask_probs = [0.5]

    for name, checkpoints_list in checkpoints.items():
        for checkpoint in checkpoints_list:
            args = deepcopy(base_args)
            args.nodes = 2
            args.name = args.name or name
            checkpoint = Path(checkpoint)

            param_sweeps = [
                (
                    f"ckpt{checkpoint.name}.lr{lr}.mlen{mlen}.mprob{mprob}.do0.1",
                    {
                        "lr": lr,
                        'mask-length': mlen,
                        'mask-prob': mprob,

                        "max-update": max_update,
                        "warmup-steps": int(max_update * 0.15),
                        "hold-steps": int(max_update * 0.4),
                        "decay-steps": int(max_update * 0.45),
                        "w2v-path": checkpoint / "checkpoint_best.pt",

                        "augment-audio": False,
                        'layerdrop': 0.1,
                        'final-dropout': 0.1,
                        'dropout': 0.1,
                        'activation-dropout': 0.1,
                        'attention-dropout': 0.1,

                        # "normalize": True,
                    }
                )
                for mlen in mask_lens
                for mprob in mask_probs
                for lr in lrs
            ]
            param_sweeps = param_sweeps[:1]
            submit.run_sweeps(args, base_params, param_sweeps, dataset='lab.10h', skip_if_cp_exists=True)
        break

if __name__ == '__main__':
    parser = submit.create_parser()
    base_args = parser.parse_args()
    sweep_func = submit.get_sweep_func(base_args.sweep)
    sweep_func(base_args)
