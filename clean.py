
import sys
from pathlib import Path

assert len(sys.argv) == 2

root = Path(sys.argv[1])
for fl in root.glob('checkpoint[0-9]*.pt'):
    print(fl)
    fl.unlink()

