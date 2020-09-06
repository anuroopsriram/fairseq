import argparse
import subprocess
from copy import deepcopy
from pathlib import Path

import submitit
import os


def create_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('name', type=str)
    parser.add_argument('sweep', type=str)
    parser.add_argument('-N', '--name', type=str, default=None)
    parser.add_argument('-d', '--data', type=Path, default='/checkpoint/anuroops/data/libris/')
    parser.add_argument('-l', '--logdir', type=Path, default='/checkpoint/anuroops/fairseq/wav2vec')
    parser.add_argument('-m', '--mem', type=int, default=400)
    parser.add_argument('-t', '--timeout', type=int, default=72)
    parser.add_argument('-p', '--partition', type=str, default='learnfair')
    parser.add_argument('-n', '--nodes', type=int, default=1)
    parser.add_argument('-g', '--gpus', type=int, default=8)
    parser.add_argument('-w', '--workers', type=int, default=9)
    parser.add_argument('--port', type=int, default=13359)
    parser.add_argument('--submit', action='store_true', default=False)
    return parser


def build_command(args, base_params, data_dir):
    params = deepcopy(base_params)
    params['save-dir'] = args.logdir / f'{args.name}'  # TODO: get jobid
    params['num-workers'] = args.workers
    params['distributed-world-size'] = args.nodes * args.gpus
    params['distributed-port'] = args.port
    cmd = ['python', '-u', 'train.py', f'{args.data}/{data_dir}']
    for param, value in sorted(params.items()):
        if not isinstance(value, (bool, list, tuple)):
            cmd.append(f'--{param} {value}')
        elif isinstance(value, (list, tuple)):
            cmd.append(f'--{param} "{value}"')
        elif value:
            cmd.append(f'--{param}')
    return cmd


def run(cmd):
    cmd = ' '.join(cmd)
    print(cmd)
    os.system(cmd)


def run_local(cmd, args):
    cmd = [f'python -m torch.distributed.launch --nproc_per_node={args.gpus}'] + cmd[2:]
    cmd = ' '.join(cmd)
    print(cmd)
    os.system(cmd)


def verify(params):
    if 'w2v-path' in params:
        assert Path(params['w2v-path']).exists()


def main(args, base_params, data_dir):
    verify(base_params)
    cmd = build_command(args, base_params, data_dir)
    print(' '.join(cmd))

    if args.submit:
        executor = submitit.AutoExecutor(folder=args.logdir / f'{args.name}')
        executor.update_parameters(
            name=args.name,
            mem_gb=args.mem,
            timeout_min=args.timeout * 60,
            slurm_partition=args.partition,
            nodes=args.nodes,
            cpus_per_task=(args.workers + 1),
            gpus_per_node=args.gpus,
            # tasks_per_node=1,
            tasks_per_node=args.gpus,
            slurm_constraint='volta32gb',
        )
        job = executor.submit(run, cmd)
        print('Submitted job:', job.job_id)
    else:
        run_local(cmd, args)


def run_sweeps(func, base_args, base_params, sweeps, dataset='unlab'):
    names = [name for name, _ in sweeps]
    assert len(set(names)) == len(names), f'Names not unique: {names}'

    base_args, base_params = func(base_args, base_params)
    for name, overrides in sweeps:
        args = deepcopy(base_args)
        args.name = f'{args.name}/{name}'
        params = deepcopy(base_params)
        params.update(**overrides)
        print(args.name, overrides)
        main(args, params, dataset)


sweep_funcs = {}

def register_sweep(func):
    sweep_funcs[func.__name__] = func


def get_sweep_func(name):
    return sweep_funcs[name]
