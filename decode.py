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
    return min_wer, min_params                        


def decode_kenlm(path, data, subset="dev_other", beamsz=1500):
    lm = LM["kenlm"]
    path = Path(path).resolve().absolute()
    if "," in str(path):
        target = Path(str(path).replace(",", "_"))
        path = target

    # Find best
    wer, params = find_best_params(path, "decode/kenlm_ax/dev_other/beam500_*/wer.*")
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


def w2v_base_mlp_augment_ablation():
    paths = [
        # "logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpFalse.tgtmlpFalse.bnFalse.actrelu.scale1.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab/lr2e-05.mlen10.mprob0.5.do0.1.ngram.lab.10h",
        # "logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpFalse.tgtmlpFalse.bnFalse.actrelu.scale1.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive_speed.snr-min8_snr-max15_speed-std0.1.unlab/lr1e-05.mlen10.mprob0.5.do0.1.ngram.lab.10h",
        # "logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab/lr1e-05.mlen10.mprob0.5.do0.1.ngram.lab.10h",
        # "logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive_speed.snr-min8_snr-max15_speed-std0.1.unlab/lr1e-05.mlen10.mprob0.5.do0.1.ngram.lab.10h",
        # "logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpTrue.tgtmlpFalse.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab/lr1e-05.mlen10.mprob0.5.do0.1.ngram.lab.10h",
        # "logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpTrue.tgtmlpFalse.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive_speed.snr-min8_snr-max15_speed-std0.1.unlab/lr1e-05.mlen10.mprob0.5.do0.1.ngram.lab.10h",
        # "logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab/lr1e-05.mlen10.mprob0.5.do0.1.ngram.lab.10h",
        # "logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive_speed.snr-min8_snr-max15_speed-std0.1.unlab/lr1e-05.mlen10.mprob0.5.do0.1.ngram.lab.10h",
    
        # "logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpFalse.tgtmlpFalse.bnFalse.actrelu.scale1.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab/lr5e-05.mlen10.mprob0.65.ngram.1nd.lab.10h",
        # "logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab/lr5e-05.mlen8.mprob0.65.ngram.1nd.lab.10h",
        # "logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpTrue.tgtmlpFalse.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab/lr5e-05.mlen12.mprob0.65.ngram.1nd.lab.10h",
        # "logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab/lr2e-05.mlen10.mprob0.65.ngram.1nd.lab.10h",

        # "logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpFalse.tgtmlpFalse.bnFalse.actrelu.scale1.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab/lr5e-05.mlen10.mprob0.65.ngram.1nd.lab.10h",
        # "logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab/lr5e-05.mlen10.mprob0.65.ngram.1nd.lab.10h",
        # "logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpTrue.tgtmlpFalse.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab/lr5e-05.mlen8.mprob0.65.ngram.1nd.lab.10h",
        # "logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab/lr5e-05.mlen8.mprob0.65.ngram.1nd.lab.10h",

        # "lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive_speed.snr-min8_snr-max15_speed-std0.1.unlab/lr1e-05.mlen10.mprob0.5.do0.1.ngram.lab.10h",
        # "lr0.0005.contextmlpTrue.tgtmlpFalse.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive_speed.snr-min8_snr-max15_speed-std0.1.unlab/lr1e-05.mlen10.mprob0.5.do0.1.ngram.lab.10h",
        # "lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive_speed.snr-min8_snr-max15_speed-std0.1.unlab/lr1e-05.mlen10.mprob0.5.do0.1.ngram.lab.10h",

        # "logs/w2v.base.mlp.augment.4x400.ft100/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive,speed.snr-min8_snr-max15_speed-std0.1.unlab/lr3e-05.mlen10.mprob0.45.do0.1.ngram.lab.100h",
        # "logs/w2v.base.glu.4x400.ft100/geglu.lr0.0005.unlab/lr2e-05.mlen10.mprob0.45.ngram.1nd.lab.100h",
        "logs/w2v.base.glu.4x400.ft100/geglu.lr0.001.unlab/lr2e-05.mlen10.mprob0.45.ngram.1nd.lab.100h",
    ]
    data = "/checkpoint/anuroops/data/libris/lab.960h"
    for path in paths:
        cmd = search_kenlm(path, data)
        # cmd = decode_kenlm(path, data)
        print(cmd)
        os.system(cmd + " &")
        # cmd = decode_viterbi(path, data)
        # os.system(cmd)


if __name__ == "__main__":
    w2v_base_mlp_augment_ablation()


#for d in *additive,speed*; do d2=`echo $d | sed "s/,/_/g"`; echo $d2; mv $d $d2; done 
