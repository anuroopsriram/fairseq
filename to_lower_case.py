import shutil
from pathlib import Path


def main(srcdir: Path, tgtdir: Path):
    ltrfiles = list(srcdir.glob("*.ltr"))
    dicts = list(srcdir.glob("dict.*.txt"))
    tsvfiles = list(srcdir.glob("*.tsv"))

    tgtdir.mkdir(exist_ok=True, parents=True)

    for fl in ltrfiles + dicts:
        with open(tgtdir / fl.name, "w") as f:
            print(fl.read_text().lower(), file=f)
            
    for fl in tsvfiles:
        shutil.copy2(fl, tgtdir / fl.name)


if __name__ == "__main__":
    # main(Path("data/ls960"), Path("data/ls960_lower"))
    main(Path("data/ted/ted.450h"), Path("data/ted_lower/ted.450h"))
    main(Path("data/ted/ted.10h"), Path("data/ted_lower/ted.10h"))
