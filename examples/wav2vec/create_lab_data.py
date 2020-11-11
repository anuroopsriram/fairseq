import argparse
from collections import Counter
from pathlib import Path

import soundfile


def get_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('root', metavar='DIR', type=Path, help='root directory containing flac files to index')
    parser.add_argument('listfile', metavar='DIR', type=Path, help='root directory containing flac files to index')
    parser.add_argument('dest', metavar='DEST', type=Path, help='output directory')
    parser.add_argument('name', metavar='N', type=str, help='output directory')
    parser.add_argument('--ext', default='flac', type=str, metavar='EXT', help='extension to look for')
    parser.add_argument('--seed', default=42, type=int, metavar='N', help='random seed')
    parser.add_argument('--write-dict', action='store_true', help='Create a dict file')
    return parser


def main(args):
    with open(args.listfile) as lstfl:
        dictionary = Counter()
        with open(args.dest / f"{args.name}.tsv", "w") as destfl, \
                open(args.dest / f"{args.name}.ltr", "w") as ltrfl, \
                open(args.dest / f"{args.name}.wrd", "w") as wrdfl:
            root = "/"
            # print(args.root, file=destfl)
            print(root, file=destfl)
            # for fl in args.root.rglob(f"*.{args.ext}"):
            for ln in lstfl:
                fl, _, _, text = ln.strip().split(maxsplit=3)
                fl = Path(fl)
                frames = soundfile.info(fl).frames
                text = text.upper()
                text_chars = text.strip().replace(" ", "|") + "|"
                dictionary.update(text_chars)
                text_chars = " ".join(text_chars)
                print(f"{fl.relative_to(root)}\t{frames}", file=destfl)
                print(text, file=wrdfl)
                print(text_chars, file=ltrfl)

        with open(args.dest / f"dict.ltr.txt", "w") as dictfl:
            for char, count in dictionary.most_common():
                print(f"{char} {count}", file=dictfl)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)

'''
python examples/wav2vec/create_lab_data.py /checkpoint/wav2letter/data/tedlium/lists/train.lst /checkpoint/anuroops/data/tedlium/ted.lab train --write-dict
python examples/wav2vec/create_lab_data.py /checkpoint/wav2letter/data/tedlium/lists/dev.lst.fixed /checkpoint/anuroops/data/tedlium/ted.lab dev
python examples/wav2vec/create_lab_data.py /checkpoint/wav2letter/data/tedlium/lists/test.lst.fixed /checkpoint/anuroops/data/tedlium/ted.lab test
'''