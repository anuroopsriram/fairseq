import pandas as pd
from pathlib import Path


args = [
    "contextmlp", "tgtmlp", "do", "ld", "augSrc", "augTgt", "augs", "snr-min", "snr-max", "speed-std"
]


def main(dirs):
    data = []

    for dir in dirs:
        fields = dir.name.split(".")
        # print(dir)
        fields = {arg: [field for field in fields if arg in field][0] for arg in args}
        fields = {arg: field[field.index(arg) + len(arg):] for arg, field in fields.items()}

        fl = next(dir.glob("*_0_log.out"))
        with open(fl) as f:
            ln = [ln.strip() for ln in f.readlines() if "best_wer" in ln][-1]
            fields["wer"] = ln.split()[-1]
        data.append(fields)
    
    data = pd.DataFrame(data)
    print(len(data))
    print(data.to_string(index=False), sep="\t")


if __name__ == "__main__":
    dirs = Path("logs/w2v.base.mlp.augment.2x100.ft").iterdir()
    main(dirs)
