

RAWDATA='/datasets01/TEDLIUM/10082020/audio/'
DATAROOT='/checkpoint/anuroops/data/tedlium/'

mkdir ${DATAROOT}/unlab

python examples/wav2vec/wav2vec_manifest.py ${RAWDATA}/train \
   --dest ${DATAROOT}/ted.unlab \
   --ext flac --valid-percent 0.01

