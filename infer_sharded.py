from pathlib import Path
from pprint import pprint

import numpy as np

import submit

base_params = {
    'path': '',
    'lm-weight': 0,
    'word-score': 0,
    'sil-weight': 0,
    'gen-subset': '',
    'results-path': '',

    'beam': 1500,
    'beam-threshold': 100,
    'beam-size-token': 100,

    'criterion': 'ctc',
    'labels': 'ltr',
    'lexicon': '/checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw/lexicon_ltr.lst',

    'remove-bpe': 'letter',
    # 'max-tokens': 4000000,
    'max-tokens': 1000000,
    'normalize': True,

    'shard-id': 0,
    'num-shards': 1,

    'task': 'audio_pretraining',
    'seed': 1,
    'nbest': 1,
}


def emissions(args, params):
    args.name = args.model.name
    # args.no32gb = True
    args.nodes = 1
    args.gpus = 8
    args.timeout = 2
    split = params['gen-subset']
    params.update({
        'beam': 1,
        'w2l-decoder': 'viterbi',
        'path': str(args.model / 'checkpoint_best.pt'),
        'dump-emissions': args.model / f'emissions_{split}.npy',
        'num-shards': args.shards,
    })
    return args, params


def uniform(a, b):
    return lambda: np.random.uniform(a, b)


def loguniform(a, b):
    return lambda: np.exp(np.random.uniform(np.log(a), np.log(b)))


def rand_search_params(config, nsamples, start=0):
    params_list = []
    for i in range(nsamples):
        params = {
            key: value() for key, value in config.items()
        }
        name = f'run{start+i:03d}'
        params_list.append((name, params))
    return params_list


def eval_4glm(args, params):
    args.name = args.name or 'kenlm'
    args.no32gb = True
    args.nodes = 1
    args.gpus = 1
    args.timeout = 2
    args.shards = 4
    params.update({
        'w2l-decoder': 'kenlm',
        'lm-model': '/checkpoint/abaevski/data/speech/libri/4-gram.bin',
        'path': str(args.model / 'checkpoint_best.pt'),
        'num-shards': args.shards,
    })
    return args, params


def eval_translm(args, params):
    args.name = args.name or 'translm'
    args.nodes = 1
    args.gpus = 1
    args.timeout = 3
    args.shards = 16
    params.update({
        'w2l-decoder': 'fairseqlm',
        'lm-model': '/checkpoint/abaevski/models/libri_lms/translm2/checkpoint_best.pt',
        'beam': 500,
        'path': str(args.model / 'checkpoint_best.pt'),
        'num-shards': args.shards,
    })
    return args, params


@submit.register_sweep
def dump_emissions(base_args):
    param_sweeps = []
    for split in base_args.splits:
        params = {
            'gen-subset': split,
            'dump-emissions': base_args.model / f'infer/emissions_{split}.npy',
        }
        param_sweeps.append((split, params))
    submit.run_sweeps(emissions, base_args, base_params, param_sweeps,
                      dataset='', task='infer', check_names=False)


@submit.register_sweep
def fixed_eval_4glm(base_args):
    name = f'lmwt{base_args.lm_weight}.wdsc{base_args.word_score}'
    hyperparams = {
        'lm-weight': base_args.lm_weight,
        'word-score': base_args.word_score,
    }
    param_sweeps = []
    for split in base_args.splits:
        # for name, overrides in param_sweeps_rs:
        params = {'gen-subset': split}
        params.update(**hyperparams)
        param_sweeps.append((name, params))

    submit.run_sweeps(eval_4glm, base_args, base_params, param_sweeps,
                      dataset='', task='infer', check_names=False)


@submit.register_sweep
def fixed_eval_translm(base_args):
    name = f'lmwt{base_args.lm_weight}.wdsc{base_args.word_score}'
    hyperparams = {
        'lm-weight': base_args.lm_weight,
        'word-score': base_args.word_score,
    }
    param_sweeps = []
    for split in base_args.splits:
        params = {'gen-subset': split}
        params.update(**hyperparams)
        param_sweeps.append((name, params))

    submit.run_sweeps(eval_translm, base_args, base_params, param_sweeps,
                      dataset='', task='infer', check_names=False)


@submit.register_sweep
def randsearch_eval_4glm(base_args):
    config = {
        'lm-weight': uniform(0, 4),
        'word-score': uniform(-4, 0),
    }
    param_sweeps_rs = rand_search_params(config, base_args.nsamples, base_args.start)
    param_sweeps = []
    for split in base_args.splits:
        for name, overrides in param_sweeps_rs:
            params = {'gen-subset': split}
            params.update(**overrides)
            param_sweeps.append((name, params))

    # param_sweeps = param_sweeps[:1]
    submit.run_sweeps(eval_4glm, base_args, base_params, param_sweeps,
                      dataset='', task='infer', check_names=False)


@submit.register_sweep
def randsearch_eval_translm(base_args):
    config = {
        'lm-weight': uniform(0, 4),
        'word-score': uniform(-4, 0),
    }
    param_sweeps_rs = rand_search_params(config, base_args.nsamples, base_args.start)
    param_sweeps = []
    for split in base_args.splits:
        for name, overrides in param_sweeps_rs:
            params = {'gen-subset': split}
            params.update(**overrides)
            param_sweeps.append((name, params))

    # param_sweeps = param_sweeps[:1]
    submit.run_sweeps(eval_translm, base_args, base_params, param_sweeps,
                      dataset='', task='infer', check_names=False)


if __name__ == '__main__':
    data_splits = ['dev_other', 'dev_clean', 'test_other', 'test_clean']

    parser = submit.create_parser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--nsamples', type=int, default=10)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--splits', type=str, nargs='*', default=data_splits)

    parser.add_argument('--lm_weight', type=float, required=False)
    parser.add_argument('--word_score', type=float, required=False)

    parser.set_defaults(data='/checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw/')
    base_args = parser.parse_args()

    sweep_func = submit.get_sweep_func(base_args.sweep)
    sweep_func(base_args)


'''

python infer_sharded.py randsearch_eval_4glm --model logs/w2v.conformer.400k.ft.4glm/dim512.enclyrs17.lr0.0005/lr2e-05.lab.960h --submit --nsamples 10

python infer_sharded.py randsearch_eval_4glm --model logs/w2v.conformer.400k.ft.4glm/dim512.enclyrs17.lr0.0005/lr2e-05/ --submit --nsamples 10


python infer_sharded.py randsearch_eval_translm --model logs/w2v.conformer.400k.ft.4glm/dim512.enclyrs17.lr0.0005/lr2e-05.lab.960h --submit --nsamples 10

python infer_sharded.py randsearch_eval_translm --model logs/w2v.conformer.400k.ft.4glm/dim512.enclyrs17.lr0.0005/lr2e-05/ --submit --nsamples 10

'''
