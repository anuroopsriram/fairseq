import os
from argparse import ArgumentParser
from pathlib import Path
import jiwer


def compute_wer(hypfls, reffls):
    def _read(fls, remove_last=False):
        lns = [
            list(open(fl).readlines()) for fl in fls
        ]
        lns = sum(lns, [])
        if remove_last:  # Remove the "(None-1234)"
            lns = [ln.rsplit(' ', 1)[0] for ln in lns]
        return lns

    hyps = _read(hypfls, True)
    refs = _read(reffls, True)
    # print(len(hyps), len(refs))
    # print(hyps[:5], refs[:5], sep='\n')
    # print()
    # print(hyps[-5:], refs[-5:], sep='\n')
    # exit()
    wer = jiwer.wer(refs, hyps)
    return wer


def evaluate_dir(direc):
    hypofiles = list(sorted(direc.glob('[0-9]_hypo.word*')))
    reffiles = list(sorted(direc.glob('[0-9]_ref.word*')))
    if len(hypofiles) == 0:
        return

    wer = compute_wer(hypofiles, reffiles)
    print(direc, wer)
    return wer

    # hypofile = direc / 'hypo.word.txt'
    # reffile = direc / 'ref.word.txt'
    # scorefile = direc / 'scores.txt'
    #
    # hypofilestr = " ".join([str(hf) for hf in hypofiles])
    # reffilestr = " ".join([str(rf) for rf in reffiles])
    # os.system(f'cat {hypofilestr} > {hypofile}')
    # os.system(f'cat {reffilestr} > {reffile}')
    #
    # scorer = '/private/home/abaevski/sctk-2.4.10/bin/sclite'
    # cmd = f"{scorer} -r {reffile} -h {hypofile}  -i rm -o all stdout > {scorefile}"
    # os.system(cmd)
    # # os.system(f'cat {scorefile}')
    # # print(f'CMD: {cmd}')


def evaluate(args):
    rundirs = args.dir.glob('run[0-9]*')
    direcs = [list(rundir.iterdir()) for rundir in rundirs]
    direcs = sum(direcs, [])
    for direc in direcs:
        evaluate_dir(direc)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args)
