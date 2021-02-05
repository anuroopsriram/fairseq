import argparse
import os
from copy import deepcopy
from pathlib import Path

import submitit


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
    parser.add_argument('--port', type=int, default=13349)
    parser.add_argument('--submit', action='store_true', default=False)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--no32gb', action='store_true')
    return parser


def create_flags(cmd, params):
    for param, value in sorted(params.items()):
        if not isinstance(value, (bool, list, tuple)):
            cmd.append(f'--{param} {value}')
        elif isinstance(value, (list, tuple)):
            cmd.append(f'--{param} "{value}"')
        elif value:
            cmd.append(f'--{param}')
    return cmd


def build_command(args, base_params, data_dir):
    params = deepcopy(base_params)
    params['save-dir'] = args.logdir / f'{args.name}'  # TODO: get jobid
    params['num-workers'] = args.workers
    params['distributed-world-size'] = args.nodes * args.gpus
    params['distributed-port'] = args.port
    cmd = ['python', '-u', 'train.py', f'{args.data}/{data_dir}']
    return create_flags(cmd, params)


def build_infer_command(args, base_params, data_dir):
    params = deepcopy(base_params)
    modeldir = Path(params['path']).parent
    data = params['gen-subset']
    results_path = modeldir / f'infer' / args.name / data
    params['results-path'] = results_path
    args.logdir = results_path
    cmd = ['python', '-u', 'examples/speech_recognition/infer.py', str(args.data / data_dir)]
    return create_flags(cmd, params)


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
        assert Path(params['w2v-path']).exists(), params["w2v-path"]


def main(args, base_params, data_dir, task):
    verify(base_params)
    assert task in ('train', 'infer')
    if task == 'train':
        cmd = build_command(args, base_params, data_dir)
    elif task == 'infer':
        cmd = build_infer_command(args, base_params, data_dir)
    else:
        raise ValueError("Unknown task")
    print(' '.join(cmd))

    if args.submit:
        executor = submitit.AutoExecutor(folder=args.logdir / args.name)
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
            slurm_constraint='volta32gb' if not args.no32gb else '',
        )
        if args.shards > 1:
            cmds = [[] + cmd + [f'--shard-id {i}'] for i in range(args.shards)]
            jobs = executor.map_array(run, cmds)
            print('Submitted jobs:', ', '.join([job.job_id for job in jobs]))
        else:
            job = executor.submit(run, cmd)
            print('Submitted job:', job.job_id)
    else:
        run_local(cmd, args)


def run_sweeps(base_args, base_params, sweeps, dataset='unlab',
               task='train', check_names=True, skip_if_cp_exists=False):
    if check_names:
        names = [name for name, _ in sweeps]
        assert len(set(names)) == len(names), f'Names not unique: {names}'

    # base_args, base_params = func(base_args, base_params)
    for name, overrides in sweeps:
        args = deepcopy(base_args)
        if dataset:
            name = name + '.' + dataset
        if task == 'train':
            args.name = f'{args.name}/{name}'
        else:
            args.name = name
        params = deepcopy(base_params)
        params.update(**overrides)
        print(args.name, overrides)

        print(args.logdir / args.name / "checkpoint_last.pt")
        if skip_if_cp_exists and (args.logdir / args.name / "checkpoint_last.pt").exists():
            print("Checkpoint exists. Skipping run \n")
            continue

        main(args, params, dataset, task)
        print()
        if not args.submit:
            break


sweep_funcs = {}


def register_sweep(func):
    sweep_funcs[func.__name__] = func


def get_sweep_func(name):
    return sweep_funcs[name]
