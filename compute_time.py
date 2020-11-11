import sys
from pathlib import Path
from datetime import datetime

from datetime import date, datetime, time
from backports.datetime_fromisoformat import MonkeyPatch
MonkeyPatch.patch_fromisoformat()


def main(logfl):
    with open(logfl) as f:
        start = None
        latest = None
        for ln in f:
            ln = ln.strip()
            if ln.endswith("begin training epoch 1"):
                start = ln.split("|")[0]
            elif "|" in ln:
                latest = ln.split("|")[0]
        start = datetime.fromisoformat(start.strip())
        latest = datetime.fromisoformat(latest.strip())
        diff = (latest - start).total_seconds() / 3600
        print(f"{str(logfl):120s}", start, latest, f"{diff:.2f}", sep="\t")
        return diff


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", type=str)
    args = parser.parse_args()

    times = []
    for f in Path("").glob(args.pattern):
        times.append(main(f))

    print(sum(times))
