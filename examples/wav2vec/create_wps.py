import sentencepiece as spm
from argparse import ArgumentParser
from pathlib import Path
from collections import Counter


def train(args, prefix):
    spm.SentencePieceTrainer.train(
        input=args.root / f"{args.file}.wrd", model_prefix=prefix,
        vocab_size=args.vocab * 1000
    )
    return prefix


def encode(args, prefix):
    sp = spm.SentencePieceProcessor(model_file=f"{prefix}.model")
    vocab = Counter()
    ltr_vocab = Counter()
    with open(args.root / f"{args.file}.wrd") as infl, \
         open(args.root / f"{args.file}.{args.vocab}k", "w") as outfl:
        for ln in infl:
            ln_sp = " ".join(sp.EncodeAsPieces(ln.strip())).replace("▁", "_")
            vocab.update(ln_sp.split())
            ltr_vocab.update(ln_sp.replace(" ", "|"))
            print(ln_sp, file=outfl)
    if args.write_vocab:
        with open(args.root / f"dict.{args.vocab}k.txt", "w") as vocfl:
            for tok, cnt in vocab.most_common():
                print(tok, cnt, file=vocfl)
    if args.write_ltr_vocab:
        with open(args.root / f"dict.ltr.txt", "w") as vocfl:
            for tok, cnt in ltr_vocab.most_common():
                print(tok, cnt, file=vocfl)


if __name__ == "__main__":
    import random
    random.seed(1234)

    parser = ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--file", type=str, default='train')
    parser.add_argument("--vocab", type=int, default=10)  # 10k
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--write-vocab", action='store_true')
    parser.add_argument("--write-ltr-vocab", action='store_true')
    args = parser.parse_args()

    prefix = f"{args.root}/spm.{args.vocab}k"
    if args.train:
        prefix = train(args, prefix)
    encode(args, prefix)


'''
python examples/wav2vec/create_wps.py --root /checkpoint/anuroops/data/tedlium/ted.lab --vocab 10 --file train --train --write-vocab 
python examples/wav2vec/create_wps.py --root /checkpoint/anuroops/data/tedlium/ted.lab --vocab 10 --file dev

python examples/wav2vec/create_wps.py --root /checkpoint/anuroops/data/ted.libris/ted.libris.lab.full --vocab 10 --file train --train --write-vocab --write-ltr-vocab 
python examples/wav2vec/create_wps.py --root /checkpoint/anuroops/data/ted.libris/ted.libris.lab.full --vocab 10 --file dev
'''

