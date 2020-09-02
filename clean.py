
import sys
from pathlib import Path

assert len(sys.argv) >= 2

for root in sys.argv[1:]:
    root = Path(root)
    for fl in root.rglob('checkpoint[0-9]*.pt'):
        print(fl)
        fl.unlink()

