import argparse
import os

import config
from pathlib import Path

# splits = {
#     "devother0.05": ["dev_other_0.05"],
#     "devother0.5": ["dev_other_0.5"],
#     "devother": ["dev_other"],
#     "all": ['dev_other', 'dev_clean', 'test_other', 'test_clean'],
# }

language_models = {
    "kenlm": {
        "path": "/checkpoint/abaevski/data/speech/libri/4-gram.bin",
        "automl_beam": 500,
        # "infer_beam": 1500,
        "infer_beam": 200,
        "gpus": 1,
        "jobs": 8,
        "shards": 1,

        # "automl_beam": 10,
        # "jobs": 2,
    },
    "fairseqlm": {
        "path": "/checkpoint/abaevski/models/libri_lms/translm2/checkpoint_best.pt",
        "automl_beam": 50,
        "infer_beam": 500,
        "gpus": 4,
        "jobs": 8,
        "shards": 4,
    }
}

lexicon = Path("/checkpoint/abaevski/data/speech/libri/960h/wav2vec/raw/lexicon_ltr.lst")
MAXTOKS = 4_000_000
# MAXTOKS = 1_000_000


def viterbi(args):
    # split = splits[args.splits][0]
    data = Path("data") / config.LAB_DATASETS[args.data]["val"]
    data, split = data.parent, data.name
    # if args.dictdata:
    #     dictpath = (Path("data") / config.LAB_DATASETS[args.dictdata]["val"]).parent
    # else:
    #     raise ValueError()
    dictpath = "data"

    cmd = (
        f"python -u examples/speech_recognition/infer.py {data} --gen-subset {split} --labels ltr "  # --lexicon {lexicon} "
        f"--target-dict {dictpath} "
        f"--path {args.model}/checkpoint_best.pt --results-path {args.model}/infer/{split} "
        f"--beam 1 --beam-size-token 100 --beam-threshold 100 --criterion ctc --lm-weight 0 --word-score 0 "
        f"--max-tokens {MAXTOKS} --nbest 1 --normalize --num-shards 1 --remove-bpe letter --post-process letter "
        f"--seed 1 --shard-id 0 --sil-weight 0 --task audio_pretraining --w2l-decoder viterbi "
    )
    print(cmd)
    os.system(cmd)


def emit(args):
    split = splits[args.splits][0]
    cmd = (
        f"python -u examples/speech_recognition/infer.py {args.data} --gen-subset {split} --labels ltr --lexicon {lexicon} "
        f"--path {args.model}/checkpoint_best.pt --dump-emissions {args.model}/infer/emissions_{split}.npy --results-path {args.model}/infer/{split} "
        f"--beam 1 --beam-size-token 100 --beam-threshold 100 --criterion ctc --lm-weight 0 --word-score 0 "
        f"--max-tokens 1000000 --nbest 1 --normalize --num-shards 1 --remove-bpe letter "
        f"--seed 1 --shard-id 0 --sil-weight 0 --task audio_pretraining --w2l-decoder viterbi "
    )
    print(cmd)
    os.system(cmd)


def automl(args):
    split = splits[args.splits][0]
    lmpath = language_models[args.lm]["path"]
    beam = language_models[args.lm]["automl_beam"]
    gpus = language_models[args.lm]["gpus"]
    jobs = language_models[args.lm]["jobs"]

    cmd = (
        f"python -u decode_automl.py -d {args.data} --gen-subset {split} -l {lexicon} "
        f"--log-dir {args.model}/automl_{args.lm} -e {args.model}/infer/emissions_{split}.npy "
        f"-m {MAXTOKS} --decoder {args.lm} --lm {lmpath} --beam {beam} --remove-bpe letter "
        f"--lmwt_min {args.lmwt_min} --lmwt_max {args.lmwt_max} --wrdsc_min {args.wrdsc_min} --wrdsc_max {args.wrdsc_max} "
        f"--prefix {split} -g {gpus} -j {jobs} --num-runs 128 --partition dev,learnfair "
    )
    if args.local:
        cmd += " --local "

    Path(f"{args.model}/automl_{args.lm}").mkdir(exist_ok=True)
    cmd += f" > {args.model}/automl_{args.lm}/OUT &"

    print(cmd)
    os.system(cmd)


def infer(args):
    if args.lmwt is None or args.wrdsc is None:
        raise ValueError(f"Require --lmwt and --wrdsc")

    split = splits[args.splits][0]
    lmpath = language_models[args.lm]["path"]
    beam = language_models[args.lm]["infer_beam"]
    # shards = language_models[args.lm]["shards"]
    shards = 1
    # TODO: Handle multiple shards
    # TODO: Submit job
    cmd = (
        f"python -u examples/speech_recognition/infer.py {args.data} --gen-subset {split} --labels ltr --lexicon {lexicon} "   
        f"--path {args.model}/checkpoint_best.pt --load-emissions {args.model}/infer/emissions_{split}.npy "
        f"--results-path {args.model}/infer_{args.lm}_{args.lmwt}_{args.wrdsc}/{split} "
        f"--beam {beam} --criterion ctc --lm-weight {args.lmwt} --word-score {args.wrdsc} --sil-weight 0 "
        f"--max-tokens {MAXTOKS} --nbest 1 --post-process letter "
        f"--seed 1 --task audio_pretraining --w2l-decoder {args.lm} --lm-model {lmpath} "
        f"--num-shards {shards} --shard-id 0 --normalize"
        # --beam-size-token 100 --beam-threshold 100 --normalize --remove-bpe letter
    )
    print(cmd)
    os.system(cmd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["viterbi", "emit", "automl", "infer"])
    parser.add_argument("model", type=Path)
    parser.add_argument("--lm", choices=["kenlm", "fairseqlm"], default="kenlm")
    # parser.add_argument("--splits", choices=splits.keys(), default="devother")
    parser.add_argument("--data", required=True)
    parser.add_argument("--dictdata", default=None)

    # AutoML
    parser.add_argument("--lmwt_min", type=float, default=0.)
    parser.add_argument("--lmwt_max", type=float, default=5.)
    parser.add_argument("--wrdsc_min", type=float, default=-5.)
    parser.add_argument("--wrdsc_max", type=float, default=5.)
    parser.add_argument("--local", action="store_true")

    # Infer
    parser.add_argument("--lmwt", type=float, default=None)
    parser.add_argument("--wrdsc", type=float, default=None)

    args = parser.parse_args()

    if args.task == "viterbi":
        viterbi(args)
    elif args.task == "emit":
        emit(args)
    elif args.task == "automl":
        automl(args)
    elif args.task == "infer":
        infer(args)
    else:
        raise ValueError(f"Unkown task: {args.task}")
