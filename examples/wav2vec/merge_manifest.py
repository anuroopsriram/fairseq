from pathlib import Path


def merge(sources, dest, read_root=True):
    print(sources, dest)
    with open(dest, "w") as df:
        for fl in sources:
            with open(fl) as f:
                root_dir = None
                if read_root:
                    root_dir = f.readline().strip().strip("/")
                for ln in f:
                    ln = ln.strip()
                    if root_dir:
                        ln = root_dir + "/" + ln.strip()
                    print(ln, file=df)


if __name__ == '__main__':
    # sources = (
    #     Path("/checkpoint/anuroops/data/tedlium/ted.unlab"),
    #     Path("/checkpoint/anuroops/data/libris/unlab"),
    # )
    # dest = Path("/checkpoint/anuroops/data/ted.libris/unlab")

    # dest.mkdir(exist_ok=True, parents=True)
    # files = (
    #     "train.tsv",
    #     # "valid.tsv"
    # )
    # for fl in files:
    #     merge(
    #         sources=list(src / fl for src in sources),
    #         dest=dest/fl
    #     )

    sources = (
        (
            Path("/checkpoint/anuroops/data/tedlium/ted.lab/train"),
            Path("/checkpoint/anuroops/data/libris/lab.960h/train"),
        ),
        (
            Path("/checkpoint/anuroops/data/tedlium/ted.lab/dev"),
            Path("/checkpoint/anuroops/data/libris/lab.960h/dev_other"),
        )
    )
    dest = (
        Path("/checkpoint/anuroops/data/ted.libris/lab.full/train"),
        Path("/checkpoint/anuroops/data/ted.libris/lab.full/dev"),
    )
    for srcs, dest in zip(sources, dest):
        dest.parent.mkdir(exist_ok=True, parents=True)
        merge(sources=[f"{src}.tsv" for src in srcs], dest=f"{dest}.tsv")
        merge(sources=[f"{src}.ltr" for src in srcs], dest=f"{dest}.ltr", read_root=False)
        merge(sources=[f"{src}.wrd" for src in srcs], dest=f"{dest}.wrd", read_root=False)
