import argparse
from collections import OrderedDict
import datetime
from glob import glob
import os
from pathlib import Path
import random
import shutil
import subprocess

from fairautoml.automl_loop import automl_param_sweep_loop
from fairautoml.automl_types import (
    HyperparamSearchSetting,
    ExecutionSetting,
    ResourceRequirement,
)


def get_args():
    parser = argparse.ArgumentParser("Script for launching hyperparameter sweeps")
    parser.add_argument("-d", "--data", required=True, help="path to data directory")
    parser.add_argument(
        "-p",
        "--prefix",
        required=True,
        help="save checkpoints and logs in <checkpoints-dir>/<prefix>.<save_dir_key>",
    )
    parser.add_argument(
        "--baseline-model", help="path to baseline model from which to resume training"
    )
    parser.add_argument(
        "--checkpoints-dir",
        default=os.path.join(
            "/checkpoint", os.environ["USER"], str(datetime.date.today())
        ),
        help="save checkpoints and logs in <checkpoints-dir>/<prefix>.<save_dir_key>",
    )

    parser.add_argument(
        "-g", "--num-gpus", type=int, required=True, help="number of GPUs per node"
    )
    parser.add_argument("--gpu-type", type=str, default="volta", choices=["volta"])
    parser.add_argument(
        "-n",
        "--num-nodes",
        type=int,
        default=1,
        help="number of nodes for distributed training",
    )
    parser.add_argument("--mem", default=100, type=int, help="memory to request")
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="output only a list of actions to perform without performing them",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="run locally instead of submitting remote job",
    )
    parser.add_argument("--partition", help="partition to run on", default="learnfair")
    parser.add_argument(
        "--constraint",
        metavar="CONSTRAINT",
        help='gpu constraint, if any. e.g. "volta"',
    )
    parser.add_argument(
        "--snapshot-code",
        action="store_true",
        default=False,
        help="Flag for creating a snapshot of training code while creating slurm job,"
        ' path is "./slurm_snapshot_code/<TIME_ISO_FORMAT/>:", '
        "can find time from comment of slurm job.",
    )
    parser.add_argument(
        "--tensorboard-logdir",
        default=os.path.join(
            "/checkpoint",
            os.environ["USER"],
            "tensorboard_logs",
            str(datetime.date.today()),
        ),
        help="save tensorboard logs in <tensorboard-logdir>/<prefix>.<save_dir_key>",
    )
    parser.add_argument(
        "--log-tensorboard", action="store_true", help="enable tensorboard logging"
    )

    parser.add_argument(
        "-j",
        "--num-parallel-jobs",
        required=True,
        type=int,
        help="numer of parallel jobs to run",
    )
    parser.add_argument(
        "-r", "--num-runs", required=True, type=int, help="total number of runs"
    )
    parser.add_argument("--time-limit", default='48:00:00', help="time limit in minutes")

    parser.add_argument(
        "--post-steps",
        nargs="+",
        help="additional steps to execute after the primary job is complete. "
        "this can be a file with the steps, or a string. some placeholders such as "
        "{job_dir} will be replaced",
    )
    parser.add_argument(
        "--extract-result",
        required=True,
        help="this is used to extract result by which each run is evaluated "
        "this can be a file with the steps, or a string. some placeholders such as "
        "{job_dir} will be replaced",
    )

    args = parser.parse_args()
    return args


class baseparam(object):
    def __init__(self, name, binary_flag=False, save_dir_key=None):
        self.name = name
        self.binary_flag = binary_flag
        self.save_dir_key = save_dir_key

    def get_save_dir_key(self, value):
        if self.save_dir_key is None:
            return None
        if self.binary_flag:
            return self.save_dir_key(1) if value else None
        if isinstance(value, float):
            value = round(value, 4)
        return self.save_dir_key(value)

    def get_cli_args(self, value, binary_flag=False):
        if binary_flag:
            return [self.name] if value else []
        else:
            return [self.name, value]

    def random_val(self):
        raise NotImplementedError()

    def automl_param(self):
        raise NotImplementedError()


class hyperparam(baseparam):
    """Base class for defining hyperparameters."""

    def __init__(self, name, value=None, binary_flag=False, save_dir_key=None):
        """
        Arguments:
        - name : the name of the hyperparameter (e.g., `--dropout`)
        - values : the set of values to sweep over (e.g., `[0.0, 0.1, 0.2]`)
        - binary_flag : whether the hyperparameter uses a boolean flag (e.g., `--no-save`)
        - save_dir_key : function that takes the hyperparameter value and returns the "key"
                         to be appended to the output directory name
        """
        super().__init__(name, binary_flag, save_dir_key)

        if value is None:  # syntactic sugar for binary flags
            self.current_value = True
            self.binary_flag = True
        else:
            self.current_value = value

    def random_val(self):
        return self.current_value

    def automl_param(self):
        return {"name": self.name, "type": "fixed", "value": self.current_value}


class rangeparam(baseparam):
    """Base class for defining hyperparameters."""

    def __init__(self, name, lower_bound, upper_bound, save_dir_key, log_scale=False):
        super().__init__(name, save_dir_key=save_dir_key)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.log_scale = log_scale

    def random_val(self):
        return (
            (random.random() * (self.upper_bound - self.lower_bound) + self.lower_bound)
            if isinstance(self.lower_bound, float) or isinstance(self.upper_bound, float)
            else random.randint(int(self.lower_bound), int(self.upper_bound) + 1)
        )

    def automl_param(self):
        return {
            "name": self.name,
            "type": "range",
            "bounds": [self.lower_bound, self.upper_bound],
            "log_scale": self.log_scale,
        }


class choiceparam(baseparam):
    """Base class for defining hyperparameters."""

    def __init__(self, name, choices, save_dir_key, is_ordered=False):
        super().__init__(name, save_dir_key=save_dir_key)
        self.choices = choices
        self.is_ordered = is_ordered

    def random_val(self):
        return random.choice(self.choices)

    def automl_param(self):
        return {
            "name": self.name,
            "type": "choice",
            "values": self.choices,
            "is_ordered": self.is_ordered,
        }


def main(get_grid, postprocess_hyperparams):
    args = get_args()

    grid = get_grid(args)

    if args.local or args.dry_run:
        args.num_nodes = 1
        config = OrderedDict()
        for p in grid:
            config[p.name] = p.random_val()

        # postprocess hyperparams
        postprocess_hyperparams(args, config)

        # launch training
        launch_local_train(config, args, grid)
    else:
        launch_automl(args, grid)


def copy_all_python_files(source, snapshot_main_dir, code_snapshot_hash):
    """
    Copies following files from source to destination:
        a) all *.py files at direct source location.
        b) all fairseq/*.py recursively.
    """
    os.makedirs(snapshot_main_dir, exist_ok=True)
    destination = os.path.join(snapshot_main_dir, code_snapshot_hash)
    assert not os.path.exists(destination), "Code snapshot: {0} alredy exists".format(
        code_snapshot_hash
    )
    os.makedirs(destination)
    all_pys = glob(os.path.join(source, "fairseq/**/*.py"), recursive=True) + glob(
        os.path.join(source, "*.py")
    )

    for filepath in all_pys:
        directory, filename = os.path.split(filepath)
        if directory:
            os.makedirs(os.path.join(destination, directory), exist_ok=True)
        shutil.copy2(
            os.path.join(source, filepath), os.path.join(destination, filepath)
        )
    return destination


def get_save_dir(args, config, params):
    # compute save_dir
    pd = {p.name: p for p in params}
    save_dir_key = ".".join(
        filter(
            lambda save_dir_key: save_dir_key is not None,
            [pd[k].get_save_dir_key(v) for k, v in config.items()],
        )
    )
    save_dir_key = save_dir_key.replace(",", "_")
    save_dir_key = save_dir_key.replace("(", "")
    save_dir_key = save_dir_key.replace(")", "")
    num_total_gpus = args.num_nodes * args.num_gpus
    save_dir = os.path.join(
        args.checkpoints_dir, f"{args.prefix}.{save_dir_key}.ngpu{num_total_gpus}"
    )
    return save_dir, save_dir_key


def dry_run(msg, args):
    if args.dry_run:
        print(f"| dry-run:  {msg}")
    return args.dry_run


def maybe_create_save_dir(args, save_dir):
    # create save directory if it doesn't exist
    if not os.path.exists(save_dir) and not args.dry_run:
        os.makedirs(save_dir)

        checkpoint_last = os.path.join(save_dir, "checkpoint_last.pt")
        if args.baseline_model and not os.path.exists(checkpoint_last):
            if not os.path.exists(args.baseline_model):
                raise FileNotFoundError(
                    f"Cannot find baseline model: {args.baseline_model}"
                )
            shutil.copyfile(args.baseline_model, checkpoint_last)


def construct_train_cmd(args, train_cmd, hyperparams, save_dir, save_dir_key):
    def make_step(s):
        if os.path.isfile(s):
            cmd = Path(post_step).read_text()
        else:
            cmd = s
        cmd = cmd.strip().format(job_dir=save_dir)
        return cmd

    post_cmds = []
    if args.post_steps:
        for post_step in args.post_steps:
            post_cmds.append(make_step(post_step))

    if args.num_nodes > 1:
        train_cmd.extend(
            [
                "--distributed-world-size",
                str(args.num_nodes * args.num_gpus),
                "--distributed-port",
                str(get_random_port()),
            ]
        )
    train_cmd.extend([args.data, "--save-dir", save_dir])
    if args.log_tensorboard:
        train_cmd.extend(
            [
                "--tensorboard-logdir",
                os.path.join(
                    args.tensorboard_logdir,
                    f"{args.prefix}.{save_dir_key}.ngpu{args.num_nodes * args.num_gpus}",
                ),
            ]
        )
    for n, v in hyperparams.items():
        if isinstance(v, bool):
            train_cmd.append(n)
        else:
            train_cmd.extend([n, str(v)])

    return train_cmd, post_cmds, make_step(args.extract_result)


def launch_automl(args, grid):
    destination = ""
    if args.snapshot_code:
        # Currently hash is just the current time in ISO format.
        code_snapshot_hash = datetime.datetime.now().isoformat()
        destination = copy_all_python_files(
            ".", "slurm_snapshot_code", code_snapshot_hash
        )

    def train_one(hyperparam):
        print(os.environ)
        result = launch_local_train(hyperparam, args, grid, destination)
        return {"result": result}

    hyperparam_search_setting = HyperparamSearchSetting(
        search_objective_name="result",
        hyperparam_setting=[p.automl_param() for p in grid],
        minimize_search_obj=True,
    )

    execution_setting = ExecutionSetting(
        run_name=args.prefix, total_num_runs=args.num_runs
    )

    log_dir = os.path.join(args.checkpoints_dir, f"{args.prefix}.auto_ml")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    resource_requirement = ResourceRequirement(
        num_tasks=1,
        num_parallel_jobs=args.num_parallel_jobs,
        cpu_per_node=args.num_gpus * 10,
        gpu_per_node=args.num_gpus,
        mem_gb_per_node=args.mem,
        gpu_type=args.gpu_type,
        constraint=args.constraint,
        time_limit=args.time_limit,
        partition=args.partition,
        log_dir=log_dir,
    )

    automl_param_sweep_loop(
        hyperparam_search_setting,
        execution_setting,
        resource_requirement,
        # the training function name
        train_one,
    )


def launch_local_train(hyperparam, args, params, destination=""):
    save_dir, save_dir_key = get_save_dir(args, hyperparam, params)

    maybe_create_save_dir(args, save_dir)

    # generate train command
    train_cmd, post_cmds, extract_result = construct_train_cmd(
        args,
        ["python", os.path.join(destination, "train.py")],
        hyperparam,
        save_dir,
        save_dir_key,
    )

    if args.dry_run:
        train_cmd_str = " ".join(train_cmd)
        dry_run(f"train command: {train_cmd_str}", args)
        for post_cmd in post_cmds:
            dry_run(f"post steps command: {post_cmd}", args)

    # start training
    env = os.environ.copy()
    assert args.num_nodes == 1, "distributed training cannot be combined with --local"
    if not dry_run("start training locally", args):
        if "CUDA_VISIBLE_DEVICES" not in env and args.num_gpus > 0:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(args.num_gpus)))
        train_proc = subprocess.Popen(train_cmd, env=env)
        train_proc.wait()
        for post_cmd in post_cmds:
            print(post_cmd)
            post_cmd_proc = subprocess.Popen(post_cmd, shell=True, env=env)
            post_cmd_proc.wait()
        if os.path.isfile(args.extract_result):
            env["PYTHONPATH"] = os.path.dirname(args.extract_result)
        extract_result_proc = subprocess.Popen(
            extract_result,
            shell=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, _ = extract_result_proc.communicate()
        try:
            return float(out)
        except:
            print(out)
            return float('inf')
    return None


def get_random_port():
    return random.randint(30000, 60000)
