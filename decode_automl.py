import argparse
from collections import OrderedDict
import os
import logging
import random
import sys

from examples.speech_recognition.infer import main as infer_main

from fairautoml.automl_loop import automl_param_sweep_loop
from fairautoml.automl_types import (
    HyperparamSearchSetting,
    ExecutionSetting,
    ResourceRequirement,
)

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser("Script for launching hyperparameter sweeps")
    parser.add_argument(
        "-d", "--data", required=True, help="data dir containing labels"
    )
    parser.add_argument("-s", "--log-dir", required=True, help="dir to save logs to")
    parser.add_argument("-t", "--targets", default="ltr", help="targets")
    parser.add_argument("-l", "--lexicon", required=True, help="lexicon file")
    parser.add_argument("-e", "--emissions", required=True, help="emissions file")
    parser.add_argument(
        "-m",
        "--max-tokens",
        type=int,
        required=True,
        help="max tokens used when extracting emissions",
    )
    parser.add_argument("--decoder", default="kenlm", choices=['kenlm', 'fairseqlm'], help="decoder type")
    parser.add_argument("--lm", required=True, help="language model location")
    parser.add_argument("--beam", type=int, default=250, help="beam size")
    parser.add_argument(
        "--beam-threshold", type=int, default=100, help="beam threshold"
    )

    parser.add_argument(
        "--sil-weight",
        action="store_true",
        help="also sweep on silence penalty",
    )

    parser.add_argument("--beam-tokens", type=int, default=100, help="beam tokens")
    parser.add_argument("--remove-bpe", type=str, default=None, help="bpe symbol")
    parser.add_argument("--gen-subset", type=str, default="dev_other", help="subset emissions are from")

    parser.add_argument("-p", "--prefix", required=True, help="prefix for the jobs")

    parser.add_argument(
        "-g", "--num-gpus", type=int, required=True, help="number of GPUs per node"
    )

    parser.add_argument("--mem", default=40, type=int, help="memory to request")

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
        "-j",
        "--num-parallel-jobs",
        required=True,
        type=int,
        help="numer of parallel jobs to run",
    )
    parser.add_argument(
        "-r", "--num-runs", required=True, type=int, help="total number of runs"
    )
    parser.add_argument(
        "--time-limit", default="48:00:00", help="time limit in minutes"
    )

    args = parser.parse_args()
    return args


class fixedparam(object):
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def random_val(self):
        return self.value

    def automl_param(self):
        return {"name": self.name, "type": "fixed", "value": self.value}


class rangeparam(object):
    def __init__(self, name, lower_bound, upper_bound, log_scale=False):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.log_scale = log_scale

    def random_val(self):
        return (
            (random.random() * (self.upper_bound - self.lower_bound) + self.lower_bound)
            if isinstance(self.lower_bound, float)
            or isinstance(self.upper_bound, float)
            else random.randint(int(self.lower_bound), int(self.upper_bound) + 1)
        )

    def automl_param(self):
        return {
            "name": self.name,
            "type": "range",
            "bounds": [self.lower_bound, self.upper_bound],
            "log_scale": self.log_scale,
        }


class choiceparam(object):
    def __init__(self, name, choices, is_ordered=False):
        self.name = name
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


def get_params(args):
    params = [
        fixedparam("data", args.data),
        fixedparam("gen_subset", args.gen_subset),
        fixedparam("max_tokens", args.max_tokens),
        fixedparam("targets", args.targets),
        fixedparam("emissions", args.emissions),
        fixedparam("decoder", args.decoder),
        fixedparam("lm", args.lm),
        fixedparam("beam", args.beam),
        fixedparam("beam_threshold", args.beam_threshold),
        fixedparam("beam_tokens", args.beam_tokens),
        rangeparam("lm_weight", 0.0, 5.0),
        rangeparam("word_score", -5.0, 5.0),
    ]
    if args.remove_bpe:
        params.append(fixedparam("remove_bpe", args.remove_bpe))
    if args.lexicon:
        params.append(fixedparam("lexicon", args.lexicon))
    if args.sil_weight:
        params.append(rangeparam("sil_weight", -10.0, 0.0))
    else:
        params.append(fixedparam("sil_weight", 0))
    return params


def main():
    args = get_args()

    if args.local:
        args.num_nodes = 1
        config = OrderedDict()

        params = get_params(args)

        for p in params:
            config[p.name] = p.random_val()

        launch_local(config)
    else:
        launch_automl(args)


def maybe_create_log_dir(args):
    # create save directory if it doesn't exist
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)


def launch_automl(args):
    hyperparam_search_setting = HyperparamSearchSetting(
        search_objective_name="wer",
        hyperparam_setting=[p.automl_param() for p in get_params(args)],
        minimize_search_obj=True,
    )

    execution_setting = ExecutionSetting(
        run_name=args.prefix, total_num_runs=args.num_runs
    )

    log_dir = os.path.join(args.log_dir, f"{args.prefix}.auto_ml")

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    resource_requirement = ResourceRequirement(
        num_tasks=1,
        num_parallel_jobs=args.num_parallel_jobs,
        cpu_per_node=max(8, args.num_gpus * 8),
        gpu_per_node=args.num_gpus,
        mem_gb_per_node=args.mem,
        gpu_type="volta",
        constraint=args.constraint,
        time_limit=args.time_limit,
        partition=args.partition,
        log_dir=log_dir,
    )

    maybe_create_log_dir(args)

    automl_param_sweep_loop(
        hyperparam_search_setting,
        execution_setting,
        resource_requirement,
        # the training function name
        launch_local,
    )

    result = hyperparam_search_setting.search_results
    best_r = None
    best_p = None
    for p, r in result:
        if best_r is None or r['wer'] < best_r:
            best_r = r['wer']
            best_p = p

    print(best_p, best_r)


def launch_local(hyperparams):
    infer_args = argparse.Namespace

    infer_args.data = hyperparams['data']
    infer_args.max_tokens = hyperparams['max_tokens']
    infer_args.w2l_decoder = hyperparams['decoder']
    infer_args.lexicon = hyperparams['lexicon'] if 'lexicon' in hyperparams else None
    infer_args.kenlm_model = hyperparams['lm']
    infer_args.load_emissions = hyperparams['emissions']
    infer_args.beam = hyperparams['beam']
    infer_args.beam_threshold = hyperparams['beam_threshold']
    infer_args.beam_size_token = hyperparams['beam_tokens']
    infer_args.lm_weight = hyperparams['lm_weight']
    infer_args.word_score = hyperparams['word_score']
    infer_args.remove_bpe = hyperparams['remove_bpe'] if 'remove_bpe' in hyperparams else None
    infer_args.gen_subset = hyperparams['gen_subset']
    infer_args.labels = hyperparams['targets']
    infer_args.sil_weight = hyperparams['sil_weight']
    infer_args.sampling = False
    infer_args.nbest = 1
    infer_args.replace_unk = None
    # infer_args.max_sentences = None
    infer_args.max_sentences = 1
    infer_args.skip_invalid_size_inputs_valid_test = False
    infer_args.required_batch_size_multiple = 8
    infer_args.num_shards = 1
    infer_args.shard_id = 0
    # infer_args.num_workers = 0
    infer_args.num_workers = 10
    infer_args.task = 'audio_pretraining'
    infer_args.criterion = 'ctc'
    infer_args.cpu = False
    infer_args.quantized = False
    infer_args.scp = False
    infer_args.sample_rate = 16000
    infer_args.max_sample_size = sys.maxsize
    infer_args.min_sample_size = 0
    infer_args.augment = False
    infer_args.lazy_load_labels = False
    infer_args.dump_emissions = False
    infer_args.dump_features = False
    infer_args.results_path = None
    infer_args.log_format = 'none'
    infer_args.log_interval = 1
    infer_args.no_progress_bar = False
    infer_args.tensorboard_logdir = None
    infer_args.prefix_size = 0
    infer_args.normalize = True
    infer_args.target_labels = None
    infer_args.unk_weight = float('-inf')
    infer_args.combine_dataset = False

    logging.disable(logging.CRITICAL)
    _, wer = infer_main(infer_args)
    logging.disable(logging.DEBUG)
    logger.info(f"{hyperparams['lm_weight']}, {hyperparams['word_score']}, {hyperparams['sil_weight']}, {wer}")
    return {"wer": wer}


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    main()
