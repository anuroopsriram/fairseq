# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import numbers
import os
import logging
from random import random

import numpy as np
import sys
import cv2

import torch
import torch.nn.functional as F

from .. import FairseqDataset

logger = logging.getLogger(__name__)


class RawAudioDataset(FairseqDataset):
    def __init__(
        self,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.sizes = []
        self.max_sample_size = (
            max_sample_size if max_sample_size is not None else sys.maxsize
        )
        self.min_sample_size = min_sample_size
        self.min_length = min_length
        self.pad = pad
        self.shuffle = shuffle
        self.normalize = normalize

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        return len(self.sizes)

    def postprocess(self, feats, curr_sample_rate):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        assert feats.dim() == 1, feats.dim()

        if self.normalize:
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
        return feats

    def crop_to_max_size(self, wav, target_size):
        size = len(wav)
        diff = size - target_size
        if diff <= 0:
            return wav

        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return wav[start:end]

    def collater(self, samples):
        samples = [
            s
            for s in samples
            if s["source"] is not None
        ]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        collated_sources, padding_mask = self._collate_sources(sources)
        input = {"source": collated_sources}

        if "target" in samples[0]:
            targets = [s["target"] for s in samples]
            collated_targets, _ = self._collate_sources(targets)
            input["target"] = collated_targets

        if self.pad:
            input["padding_mask"] = padding_mask
        return {"id": torch.LongTensor([s["id"] for s in samples]), "net_input": input}

    def _collate_sources(self, sources):
        sizes = [len(s) for s in sources]
        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)
        collated_sources = sources[0].new(len(sources), target_size)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff,), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)
        return collated_sources, padding_mask

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]


class FileAudioDataset(RawAudioDataset):
    def __init__(
        self,
        manifest_path,
        sample_rate,
        max_sample_size=None,
        min_sample_size=None,
        shuffle=True,
        min_length=0,
        pad=False,
        normalize=False,
    ):
        super().__init__(
            sample_rate=sample_rate,
            max_sample_size=max_sample_size,
            min_sample_size=min_sample_size,
            shuffle=shuffle,
            min_length=min_length,
            pad=pad,
            normalize=normalize,
        )

        self.fnames = []

        skipped = 0
        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for line in f:
                items = line.strip().split("\t")
                assert len(items) == 2, line
                sz = int(items[1])
                if min_length is not None and sz < min_length:
                    skipped += 1
                    continue
                self.fnames.append(items[0])
                self.sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

    def __getitem__(self, index):
        import soundfile as sf

        fname = os.path.join(self.root_dir, self.fnames[index])
        wav, curr_sample_rate = sf.read(fname)
        feats = torch.from_numpy(wav).float()
        feats = self.postprocess(feats, curr_sample_rate)
        return {"id": index, "source": feats}


def calc_mean_invstddev(feature, eps = 1e-8):
    if len(feature.size()) != 2:
        raise ValueError("We expect the input feature to be 2-D tensor")
    mean = feature.mean(0)
    var = feature.var(0)
    # avoid division by ~zero
    # if (var < eps).any():
    #     return mean, 1.0 / (torch.sqrt(var) + eps)
    # return mean, 1.0 / torch.sqrt(var)
    return mean, 1.0 / (torch.sqrt(var) + eps)


def apply_mv_norm(features):
    # If there is less than 2 spectrograms, the variance cannot be computed (is NaN)
    # and normalization is not possible, so return the item as it is
    if features.size(0) < 2:
        return features
    mean, invstddev = calc_mean_invstddev(features)
    res = (features - mean) * invstddev
    return res


class LogMelAudioDataset(FileAudioDataset):
    def __init__(
            self,
            manifest_path,
            sample_rate,
            max_sample_size=None,
            min_sample_size=None,
            shuffle=True,
            min_length=0,
            pad=False,
            num_mel_bins=80,
            frame_length=25.0,
            frame_shift=10.0,
            specaug_prob=0.
    ):
        super().__init__(
            manifest_path,
            sample_rate,
            max_sample_size,
            min_sample_size,
            shuffle,
            min_length,
            pad,
            normalize=False)
        self.num_mel_bins = num_mel_bins
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.transform = None
        self.specaug_prob = specaug_prob
        if specaug_prob > 0:
            self.transform = SpecAugmentTransform()

    def postprocess(self, feats, curr_sample_rate):
        if feats.dim() == 2:
            feats = feats.mean(-1)

        if curr_sample_rate != self.sample_rate:
            raise Exception(f"sample rate: {curr_sample_rate}, need {self.sample_rate}")

        assert feats.dim() == 1, feats.dim()

        # Compute MEL
        # import torchaudio.compliance.kaldi as kaldi
        # fbank = kaldi.fbank(
        #     feats.unsqueeze(0),
        #     sample_frequency=curr_sample_rate,
        #     num_mel_bins=self.num_mel_bins,
        #     frame_length=self.frame_length,
        #     frame_shift=self.frame_shift,
        # )

        from python_speech_features.base import logfbank
        fbank = logfbank(
            feats.numpy(),
            samplerate=self.sample_rate,
            winlen=self.frame_length / 1000.,
            winstep=self.frame_shift / 1000.,
            nfilt=self.num_mel_bins,
            lowfreq=20.,
            wintype='povey',
            dither=0.
        ).astype(np.float32)
        fbank = feats.new(fbank)
        if self.transform is not None:
            if random() < self.specaug_prob:
                fbank = self.transform(fbank.numpy())
                fbank = feats.new(fbank)
        fbank = apply_mv_norm(fbank)
        return fbank

    def crop_to_max_size(self, wav, target_size):
        size = wav.shape[0]
        diff = size - target_size
        if diff <= 0:
            return wav

        start = np.random.randint(0, diff + 1)
        end = size - diff + start
        return wav[start:end]

    def collater(self, samples):
        samples = [
            s
            for s in samples
            if s["source"] is not None
        ]
        if len(samples) == 0:
            return {}

        sources = [s["source"] for s in samples]
        sizes = [s.shape[0] for s in sources]

        if self.pad:
            target_size = min(max(sizes), self.max_sample_size)
        else:
            target_size = min(min(sizes), self.max_sample_size)

        collated_sources = sources[0].new(len(sources), target_size, self.num_mel_bins)
        padding_mask = (
            torch.BoolTensor(collated_sources.shape).fill_(False) if self.pad else None
        )
        for i, (source, size) in enumerate(zip(sources, sizes)):
            diff = size - target_size
            if diff == 0:
                collated_sources[i] = source
            elif diff < 0:
                assert self.pad
                collated_sources[i] = torch.cat(
                    [source, source.new_full((-diff, self.num_mel_bins), 0.0)]
                )
                padding_mask[i, diff:] = True
            else:
                collated_sources[i] = self.crop_to_max_size(source, target_size)

        input = {"source": collated_sources}
        if self.pad:
            input["padding_mask"] = padding_mask
        return {"id": torch.LongTensor([s["id"] for s in samples]), "net_input": input}


class SpecAugmentTransform:
    """
    Implementation of SpecAugment transform for acoustic features.
    See ref: https://arxiv.org/pdf/1904.08779.pdf
    """

    def __init__(
            self,
            time_warp_W=0,
            freq_mask_N=1,
            freq_mask_F=27,
            time_mask_N=10,
            time_mask_T=10000,
            time_mask_p=0.05,
            mask_value=None,
            add_noise=False,
    ):
        # Sanity checks
        assert mask_value is None or isinstance(
            mask_value, numbers.Number
        ), "mask_value (type: {}) must be None or a number".format(type(mask_value))
        if freq_mask_N > 0:
            assert (
                    freq_mask_F > 0
            ), "freq_mask_F ({}) must be larger than 0 when doing freq masking.".format(
                freq_mask_F
            )
        if time_mask_N > 0:
            assert (
                    time_mask_T > 0
            ), "time_mask_T ({}) must be larger than 0 when doing time masking.".format(
                time_mask_T
            )

        self.time_warp_W = time_warp_W
        self.freq_mask_N = freq_mask_N
        self.freq_mask_F = freq_mask_F
        self.time_mask_N = time_mask_N
        self.time_mask_T = time_mask_T
        self.time_mask_p = time_mask_p
        self.mask_value = mask_value if mask_value is None else float(mask_value)
        self.add_noise = add_noise

        if mask_value is None:
            logger.info("mask_value is None, will use local mean for masking")

    def __call__(self, spectrogram):
        assert len(spectrogram.shape) == 2, "spectrogram must be a 2-D tensor."

        distorted = spectrogram.copy()  # make a copy of input spectrogram.
        num_frames = spectrogram.shape[0]  # or 'tau' in the paper.
        num_freqs = spectrogram.shape[1]  # or 'miu' in the paper.
        mask_value = self.mask_value

        if mask_value is None:  # if no value was specified, use local mean.
            mask_value = spectrogram.mean()

        if num_frames == 0:
            logger.warning("Empty input data, ignoring this sequence")
            return spectrogram

        if num_freqs < self.freq_mask_F:
            if self.freq_mask_N > 0:
                # If freq masking is on, this should not happen and need double check
                # on feature dimension.
                logger.warning(
                    "Input sequence has {} spectrums while maximum ".format(num_freqs)
                    + "{} spectrum will be distorted.".format(self.freq_mask_F)
                    + " Ignoring this sequence."
                )
            # or else, freq masking is being off
            return spectrogram

        if self.time_warp_W > 0:
            if 2 * self.time_warp_W >= num_frames:
                logger.warning(
                    "Time warpping skipped since 2*time_warp_W >= frame size"
                )
            else:
                w0 = np.random.randint(self.time_warp_W, num_frames - self.time_warp_W)
                w = np.random.randint(0, self.time_warp_W)
                upper, lower = distorted[:w0, :], distorted[w0:, :]
                upper = cv2.resize(
                    upper, dsize=(num_freqs, w0 + w), interpolation=cv2.INTER_LINEAR
                )
                lower = cv2.resize(
                    lower,
                    dsize=(num_freqs, num_frames - w0 - w),
                    interpolation=cv2.INTER_LINEAR,
                )
                distorted = np.concatenate((upper, lower), axis=0)

        for _i in range(self.freq_mask_N):
            f = np.random.randint(0, self.freq_mask_F)
            f0 = np.random.randint(0, num_freqs - f)
            if f != 0:
                distorted[:, f0: f0 + f] = mask_value

        max_time_mask_T = min(
            self.time_mask_T, math.floor(num_frames * self.time_mask_p)
        )
        if max_time_mask_T < 1:
            if self.time_mask_N > 0:  # If time masking is on
                logger.warning(
                    "Input sequence has {} frames while maximum ".format(num_frames)
                    + "{} frames will be distorted.".format(max_time_mask_T)
                    + " Ignoring time masking."
                )
            # or else time masking is off
            return distorted

        # time_mask_N = np.random.randint(0, self.time_mask_N)
        time_mask_N = self.time_mask_N
        for _i in range(time_mask_N):
            t = np.random.randint(0, max_time_mask_T)
            t0 = np.random.randint(0, num_frames - t)
            if t != 0:
                distorted[t0: t0 + t, :] = mask_value
                if self.add_noise:
                    distorted[t0: t0 + t, :] += np.random.normal(
                        0, 1, size=(t, distorted.shape[1])
                    )

        return distorted
