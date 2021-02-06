import shutil
from pathlib import Path


def merge(datasets, dest):
    dest.mkdir(exist_ok=True)
    with open(dest / "train.tsv", "w") as destf:
        print("", file=destf)
        for _, src in datasets.items():
            with open(src / "train.tsv") as srcf:
                srcroot = Path(srcf.readline().strip())
                for ln in srcf.readlines():
                    path, num = ln.strip().split()
                    print(srcroot / path, num, sep="\t", file=destf)
        
    for name, src in datasets.items():
        srcfl: Path = src / "valid.tsv"
        shutil.copy2(srcfl, dest / f"valid_{name}.tsv")

    # for name, src in datasets.items():
    #     srcfl: Path = src / "train.tsv"
    #     shutil.copy2(srcfl, dest / f"train_{name}.tsv")


if __name__ == "__main__":
    datasets = {
        "lv": Path("/private/home/abaevski/data/librivox/no_silence"),
        "fsh.swbd": Path("/checkpoint/anuroops/data/asr/fsh.swb/fsh.swb.unlab"),
    }
    merge(datasets, Path("/checkpoint/anuroops/data/asr/lv.fsh.swb"))
