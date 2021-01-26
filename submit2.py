import argparse
import os
from copy import deepcopy
from pathlib import Path

import submitit


def create_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('name', type=str)
    # parser.add_argument('sweep', type=str)
    parser.add_argument('-l', '--logdir', type=Path, default='exp')
    parser.add_argument('-N', '--name', type=str, default=None)
    parser.add_argument('-d', '--config-dir', type=Path, default="examples/wav2vec/config/pretraining")
    parser.add_argument('-m', '--mem', type=int, default=400)
    parser.add_argument('-t', '--timeout', type=int, default=72)
    parser.add_argument('-p', '--partition', type=str, default='learnfair')
    parser.add_argument('-n', '--nodes', type=int, default=1)
    parser.add_argument('-g', '--gpus', type=int, default=8)
    parser.add_argument('-w', '--workers', type=int, default=9)
    parser.add_argument('--submit', action='store_true', default=False)
    return parser


def create_flags(cmd, params):
    for param, value in sorted(params.items()):
        cmd.append(f"{param}={value}")
    return cmd


def build_command(args, base_config, overrides):
    cmd = ["fairseq-hydra-train"]
    overrides = deepcopy(overrides)
    overrides["distributed_training.distributed_world_size"] = args.nodes * args.gpus
    overrides["checkpoint.save_dir"] = args.logdir
    cmd = create_flags(cmd, overrides)
    cmd.extend([f"--config-dir={args.config_dir}", f"--config-name={base_config}"])    
    return cmd


def run(cmd):
    cmd = ' '.join(cmd)
    print(cmd)
    os.system(cmd)


def main(args, base_config, overrides):
    print()

    cmd = build_command(args, base_config, overrides)
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
            tasks_per_node=args.gpus,
            slurm_constraint='volta32gb',
        )
        job = executor.submit(run, cmd)
        print('Submitted job:', job.job_id)
    else:
        run(cmd)
