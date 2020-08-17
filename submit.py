import argparse
import subprocess
from copy import deepcopy
from pathlib import Path

import submitit


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
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
    # proc = subprocess.Popen(cmd)
    # proc.wait()
    import os
    cmd = ' '.join(cmd)
    print(cmd)
    os.system(cmd)


def main(args, base_params, data_dir):
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
        run(cmd)
