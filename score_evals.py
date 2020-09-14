import os
from argparse import ArgumentParser
from pathlib import Path
import jiwer
import multiprocessing as mp

counts = {
    'dev_other': 2864,
    'dev_clean': 2703,
    'test_other': 2939,
    'test_clean': 2620,
}


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

    num = 0
    name = str(hypfls[0])
    for key, value in counts.items():
        if key in name:
            num = value
            break

    # assert 0 < num == len(hyps) and len(refs) == len(hyps)
    if not (0 < num == len(hyps) and len(refs) == len(hyps)):
        print('ERR', key, hypfls[0].parent, num, len(hyps), len(refs))
        return -1

    wer = jiwer.wer(refs, hyps)
    return wer


def evaluate_dir(direc):
    print('Running', direc)

    hypofiles = list(sorted(direc.glob('[0-9]_hypo.word*')))
    reffiles = list(sorted(direc.glob('[0-9]_ref.word*')))
    if len(hypofiles) == 0:
        return

    wer = compute_wer(hypofiles, reffiles)
    if wer < 0:
        return wer

    with open(direc / 'score.txt', 'w') as fl:
        print(direc, wer, file=fl)
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
    with mp.Pool(1) as pool:
        wers = pool.map(evaluate_dir, direcs)
    print('\n'.join(wers))

    # wers = []
    # for direc in direcs:
    #     wer = evaluate_dir(direc)
    #     wers.append((wer, direc))
    #     print(wer, direc)
    # print('\n'.join(wers))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args)
