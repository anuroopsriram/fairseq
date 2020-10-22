import sys
from pathlib import Path
from datetime import datetime


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
        print(start, latest, diff)


if __name__ == '__main__':
    logfl = Path(sys.argv[1])
    main(logfl)
