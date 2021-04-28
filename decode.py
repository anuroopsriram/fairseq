import argparse
import os
import re
from pathlib import Path

from submit import main


LM = {
    "kenlm": "ls_4gram",
}


def search_kenlm(path, data, subset="dev_other", beamsz=500):
    lm = LM["kenlm"]
    path = Path(path).resolve().absolute()
    if "," in str(path):
        target = Path(str(path).replace(",", "_"))
        path = target

    cmd = f"""
    python examples/speech_recognition/hydra/infer.py --multirun 
        --config-dir=examples/speech_recognition/hydra/conf/ 
        --config-name=infer_kenlm 
        hydra/launcher=submitit_slurm 
        hydra/sweeper=ax 
        +run=slurm1 
        +lm={lm}
        +ax_sweep=ngram1 
        task=audio_pretraining 
        task.data={data}
        task.labels=ltr 
        decoding.decoder.beam={beamsz}
        decoding.exp_dir="{path}"
        decoding.write_sentences=false 
        decoding.unique_wer_file=true 
        dataset.gen_subset={subset}
        dataset.max_tokens=1100000 
        common_eval.path={path}/checkpoint_best.pt
    """
    cmd = cmd.replace("\n", " ")
    cmd = re.sub(" +", " ", cmd)
    return cmd


def decode_viterbi(path, data, subset="dev_other"):
    path = Path(path).resolve().absolute()
    cmd = f"""
    python examples/speech_recognition/hydra/infer.py \
        --config-dir=examples/speech_recognition/hydra/conf/ \
        --config-name=infer_viterbi \
        task=audio_pretraining \
        task.data={data} \
        task.labels=ltr \
        decoding.exp_dir="{path}" \
        decoding.write_sentences=true \
        decoding.unique_wer_file=true \
        dataset.gen_subset={subset} \
        dataset.max_tokens=1100000 \
        common_eval.path={path}/checkpoint_best.pt
    """
    cmd = cmd.replace("\n", " ")
    cmd = re.sub(" +", " ", cmd)
    return cmd


def find_best_params(path, glob):
    import yaml
    
    min_wer, min_params = 1000, {}
    count = len(list(path.glob(glob)))
    opt_exists = ((path / glob).parent.parent / "optimization_results.yaml").exists()
    for wer_file in path.glob(glob):
        with open(wer_file) as wf:
            contents = wf.readlines()
            contents = contents[:1] + contents[2:]
            conf = yaml.safe_load("\n".join(contents))
            wer = conf["WER"]
            if wer < min_wer:
                min_wer = wer
                min_params = {
                    "lmweight": conf["decoder"]["lmweight"],
                    "wordscore": conf["decoder"]["wordscore"],
                    "silweight": conf["decoder"]["silweight"],
                }
    return min_wer, min_params, count, opt_exists


def decode_kenlm(path, data, subset="dev_other", beamsz=1500):
    lm = LM["kenlm"]
    path = Path(path).resolve().absolute()
    if "," in str(path):
        target = Path(str(path).replace(",", "_"))
        path = target

    # Find best
    wer, params, _, _ = find_best_params(path, "decode/kenlm_ax/dev_other/beam500_*/wer.*")
    print(wer, params)
    cmd = f"""
    python examples/speech_recognition/hydra/infer.py --multirun 
        --config-dir=examples/speech_recognition/hydra/conf/ 
        --config-name=infer_kenlm 
        hydra/launcher=submitit_slurm 
        +run=slurm1 
        +lm={lm}
        task=audio_pretraining 
        task.data={data}
        task.labels=ltr 
        decoding.decoder.beam={beamsz}
        decoding.exp_dir="{path}"
        decoding.write_sentences=true 
        decoding.unique_wer_file=true 
        decoding.decoder.lmweight={params["lmweight"]} \
        decoding.decoder.wordscore={params["wordscore"]} \
        decoding.decoder.silweight={params["silweight"]} \
        dataset.gen_subset={subset}
        dataset.max_tokens=1100000 
        common_eval.path={path}/checkpoint_best.pt
    """
    cmd = cmd.replace("\n", " ")
    cmd = re.sub(" +", " ", cmd)
    return cmd


def main(args):
    paths = [
        # "logs/ablation.aug.ls960h.3x400.ft/ls960h.add8.15lr0.0005.ls960h/lab.1h.lr3e-05.mlen10.mprob0.65.do0.1.lab.1h",
        # "logs/ablation.aug.ls960h.3x400.ft/ls960h.noauglr0.0005.ls960h/lab.1h.lr5e-05.mlen10.mprob0.65.do0.1.lab.1h",

        # "logs/ablation.aug.ls960h.3x400.ft/ls960h.add8.15lr0.0005.ls960h/lab.100h.lr3e-05.mlen10.mprob0.45.do0.1.lab.100h",
        # "logs/ablation.aug.ls960h.3x400.ft/ls960h.noauglr0.0005.ls960h/lab.100h.lr3e-05.mlen10.mprob0.65.do0.1.lab.100h",

        # "logs/ablation.mlp.ls960h.3x400.ft/lr0.0005.cmlpTrue.tmlpTrue.bnTrue.actrelu.scale4.nhid0.unlab/lab.100h.lr3e-05.mlen10.mprob0.45.do0.1.lab.100h",
        # "logs/ablation.mlp.ls960h.3x400.ft/lr0.0005.cmlpTrue.tmlpTrue.bnTrue.actrelu.scale4.nhid1.unlab/lab.100h.lr3e-05.mlen10.mprob0.45.do0.1.lab.100h",
        # "logs/ablation.mlp.ls960h.3x400.ft/lr0.0005.cmlpTrue.tmlpTrue.bnTrue.actrelu.scale4.nhid2.unlab/lab.100h.lr3e-05.mlen10.mprob0.45.do0.1.lab.100h",
        # "logs/ablation.mlp.ls960h.3x400.ft/lr0.0005.cmlpTrue.tmlpTrue.bnTrue.actrelu.scale4.nhid0.unlab/lab.1h.lr3e-05.mlen10.mprob0.65.do0.1.lab.1h",
        # "logs/ablation.mlp.ls960h.3x400.ft/lr0.0005.cmlpTrue.tmlpTrue.bnTrue.actrelu.scale4.nhid1.unlab/lab.1h.lr3e-05.mlen10.mprob0.65.do0.1.lab.1h",
        # "logs/ablation.mlp.ls960h.3x400.ft/lr0.0005.cmlpTrue.tmlpTrue.bnTrue.actrelu.scale4.nhid2.unlab/lab.1h.lr3e-05.mlen10.mprob0.65.do0.1.lab.1h",

        # "logs/ablation.aug.ls100h.3x200.ft/ls100h.add8.15lr0.0005.ls100h/lab.1h.lr3e-05.mlen10.mprob0.65.do0.1.lab.1h",
        # "logs/ablation.aug.ls100h.3x200.ft/ls100h.noauglr0.0005.ls100h/lab.1h.lr3e-05.mlen10.mprob0.65.do0.1.lab.1h",
        # "logs/ablation.aug.ls100h.3x200.ft/ls100h.add8.15lr0.0005.ls100h/lab.100h.lr3e-05.mlen10.mprob0.65.do0.1.lab.100h",
        # "logs/ablation.aug.ls100h.3x200.ft/ls100h.noauglr0.0005.ls100h/lab.100h.lr3e-05.mlen10.mprob0.65.do0.1.lab.100h",

        # "logs/ablation.aug.ls400h.3x300.ft/ls400h.add8.15lr0.0005.ls400h/lab.100h.lr3e-05.mlen10.mprob0.65.do0.1.lab.100h",
        # "logs/ablation.aug.ls400h.3x300.ft/ls400h.noauglr0.0005.ls400h/lab.100h.lr5e-05.mlen10.mprob0.65.do0.1.lab.100h",
        # "logs/ablation.aug.ls400h.3x300.ft/ls400h.add8.15lr0.0005.ls400h/lab.1h.lr3e-05.mlen10.mprob0.65.do0.1.lab.1h",
        # "logs/ablation.aug.ls400h.3x300.ft/ls400h.noauglr0.0005.ls400h/lab.1h.lr5e-05.mlen10.mprob0.65.do0.1.lab.1h",

        # "logs/ablation.baseline.ls960h.8x400.ft/ls960h.baselinelr0.0005.ls960h/lab.1h.lr3e-05.mlen10.mprob0.65.do0.1.lab.1h",
        # "logs/ablation.baseline.ls960h.8x400.ft/ls960h.baselinelr0.0005.ls960h/lab.100h.lr3e-05.mlen10.mprob0.45.do0.1.lab.100h",

        # "logs/ablation.conf.ls960h.3x400.ft/ls960h.conf.lr0.0005.ks3.normbatchnorm.ls960h/lab.1h.lr3e-05.mlen10.mprob0.65.do0.1.lab.1h",
        # "logs/ablation.conf.ls960h.3x400.ft/ls960h.conf.lr0.0005.ks3.normlayernorm.ls960h/lab.100h.lr3e-05.mlen10.mprob0.65.do0.1.lab.100h",
        # "logs/ablation.conf.ls960h.3x400.ft/ls960h.conf_rp.lr0.0005.ks3.normlayernorm.ls960h/lab.1h.lr5e-05.mlen10.mprob0.45.do0.1.lab.1h",

        # "logs/ablation.aug.ls50h.3x200.ft/ls50h.noauglr0.0005.ls50h/lab.1h.lr3e-05.mlen10.mprob0.45.do0.1.lab.1h",
        # "logs/ablation.aug.ls50h.3x200.ft/ls50h.add8.15lr0.0005.ls50h/lab.1h.lr3e-05.mlen10.mprob0.45.do0.1.lab.1h",
        # "logs/ablation.aug.ls50h.3x200.ft/ls50h.add8.15lr0.0005.ls50h/lab.100h.lr3e-05.mlen10.mprob0.65.do0.1.lab.100h",
        # "logs/ablation.aug.ls50h.3x200.ft/ls50h.noauglr0.0005.ls50h/lab.100h.lr5e-05.mlen10.mprob0.65.do0.1.lab.100h",

        # "logs/ablation.aug.ls100h.3x200.ft/ls100h.sameaug.add8.15lr0.0005.ls100h/lab.1h.lr3e-05.mlen10.mprob0.65.do0.1.lab.1h",
        # "logs/ablation.aug.ls100h.3x200.ft/ls100h.sameaug.add8.15lr0.0005.ls100h/lab.100h.lr3e-05.mlen10.mprob0.65.do0.1.lab.100h",

        # "logs/ablation.lconv.ls960h.3x400.ft/ls960h.lc_last2.lr0.0005.moddo0.1.ls960h/lab.100h.lr3e-05.mlen10.mprob0.65.do0.1.lab.100h",
        # "logs/ablation.lconv.ls960h.3x400.ft/ls960h.dc_last2.lr0.0005.moddo0.1.ls960h/lab.100h.lr3e-05.mlen10.mprob0.45.do0.1.lab.100h",
        # "logs/ablation.lconv.ls960h.3x400.ft/ls960h.dc_last2.lr0.0005.moddo0.1.ls960h/lab.1h.lr3e-05.mlen10.mprob0.65.do0.1.lab.1h",
        # "logs/ablation.lconv.ls960h.3x400.ft/ls960h.lc_last2.lr0.0005.moddefault.ls960h/lab.1h.lr3e-05.mlen10.mprob0.65.do0.1.lab.1h",
    ]
    data = "/checkpoint/anuroops/data/libris/lab.960h"
    for path in paths:
        if args.task == "search":
            cmd = search_kenlm(path, data)
            os.system(cmd + " &")
        elif args.task == "viterbi":
            cmd = decode_viterbi(path, data)
            os.system(cmd)
        elif args.task == "beam":
            cmd = decode_kenlm(path, data)
            os.system(cmd + " &")
        elif args.task == "searchres":
            wer, params, count, opt_exists = find_best_params(Path(path), "decode/kenlm_ax/dev_other/beam500_*/wer.*")
            print(path)
            print(wer, count, opt_exists, params)
        elif args.task == "beamres":
            wer, params, _, _ = find_best_params(Path(path), "decode/kenlm.1500/dev_other/beam1500_*/wer.*")
            print(path)
            print(wer, params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=["search", "viterbi", "beam", "searchres", "beamres"])
    args = parser.parse_args()
    main(args)
