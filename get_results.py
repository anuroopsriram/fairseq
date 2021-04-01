import pandas as pd
from pathlib import Path


def get_wer(dir):
    fls = list(dir.glob("*_0_log.out"))
    if len(fls) == 0:
        return 1e6
    fl = sorted(fls)[-1]
    with open(fl) as f:
        lns = [ln.strip() for ln in f.readlines() if "best_wer" in ln]
        if len(lns) == 0:
            return 1e6
        wer = float(lns[-1].split()[-1])
    return wer


# args = [
#     "contextmlp", "tgtmlp", "do", "ld", "augSrc", "augTgt", "augs", "snr-min", "snr-max", "speed-std"
# ]


# def main(dirs):
#     data = []

#     for dir in dirs:
#         fields = dir.name.split(".")
#         # print(dir)
#         fields = {arg: [field for field in fields if arg in field][0] for arg in args}
#         fields = {arg: field[field.index(arg) + len(arg):] for arg, field in fields.items()}

#         fl = next(dir.glob("*_0_log.out"))
#         with open(fl) as f:
#             ln = [ln.strip() for ln in f.readlines() if "best_wer" in ln][-1]
#             fields["wer"] = ln.split()[-1]
#         data.append(fields)
    
#     data = pd.DataFrame(data)
#     print(len(data))
#     print(data.to_string(index=False), sep="\t")


def main(dirs):
    res = {}
    for dir in dirs:
        print(dir)
        res_dir = {}
        for dir2 in dir.iterdir():
            wer = get_wer(dir2)
            if wer == 1e6:
                continue
            res_dir[dir2] = wer
            print(dir2, wer)
            for dir3 in dir2.iterdir():
                wer = get_wer(dir3)
                if wer == 1e6:
                    continue
                res_dir[dir3] = wer
                print(dir3, wer)
        res_dir = sorted(list(res_dir.items()), key=lambda x: x[1])
        res[dir] = res_dir[0]
        print()
    for dir, (dir2, wer) in res.items():
        print(dir, dir2, wer)


if __name__ == "__main__":
    # dirs = Path("logs/w2v.base.mlp.augment.2x100.ft").iterdir()
    dirs = Path("logs/w2v.base.mlp.augment.4x400.ft").iterdir()
    main(dirs)


'''
logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpFalse.tgtmlpFalse.bnFalse.actrelu.scale1.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpFalse.tgtmlpFalse.bnFalse.actrelu.scale1.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab/lr5e-05.mlen10.mprob0.65.ngram.1nd.lab.10h 10.597

logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpFalse.tgtmlpFalse.bnFalse.actrelu.scale1.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive_speed.snr-min8_snr-max15_speed-std0.1.unlab logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpFalse.tgtmlpFalse.bnFalse.actrelu.scale1.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive_speed.snr-min8_snr-max15_speed-std0.1.unlab/lr1e-05.mlen10.mprob0.5.do0.1.ngram.lab.10h 10.542

logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab/lr5e-05.mlen8.mprob0.65.ngram.1nd.lab.10h 10.784

logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpTrue.tgtmlpFalse.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpTrue.tgtmlpFalse.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab/lr5e-05.mlen12.mprob0.65.ngram.1nd.lab.10h 10.57

logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc0.0.augTgt0.0.augsadditive.speed-std0.1.unlab/lr2e-05.mlen10.mprob0.65.ngram.1nd.lab.10h 10.542

logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive_speed.snr-min8_snr-max15_speed-std0.1.unlab logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpFalse.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive_speed.snr-min8_snr-max15_speed-std0.1.unlab/lr1e-05.mlen10.mprob0.5.do0.1.ngram.lab.10h 10.556

logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpTrue.tgtmlpFalse.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive_speed.snr-min8_snr-max15_speed-std0.1.unlab logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpTrue.tgtmlpFalse.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive_speed.snr-min8_snr-max15_speed-std0.1.unlab/lr1e-05.mlen10.mprob0.5.do0.1.ngram.lab.10h 10.793

logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive_speed.snr-min8_snr-max15_speed-std0.1.unlab logs/w2v.base.mlp.augment.4x400.ft/lr0.0005.contextmlpTrue.tgtmlpTrue.bnTrue.actrelu.scale4.do0.1.ld0.05.normFalseaugSrc1.0.augTgt1.0.augsadditive_speed.snr-min8_snr-max15_speed-std0.1.unlab/lr1e-05.mlen10.mprob0.5.do0.1.ngram.lab.10h 10.821
'''
