# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import librosa
import numpy as np
import torch
from pathlib import Path
import augment
from torch.nn import functional as F

from .. import BaseWrapperDataset


class ChainRunner:
    """
    Takes an instance of augment.EffectChain and applies it on pytorch tensors.
    """

    def __init__(self, chain):
        self.chain = chain

    def __call__(self, x):
        """
        x: torch.Tensor, (channels, length). Must be placed on CPU.
        """
        x = x.view(1, -1)
        src_info = {'channels': x.size(0),  # number of channels
                    'length': x.size(1),  # length of the sequence
                    'precision': 32,  # precision (16, 32 bits)
                    'rate': 16000.0,  # sampling rate
                    'bits_per_sample': 32}  # size of the sample

        target_info = {'channels': 1,
                       'length': x.size(1),
                       'precision': 32,
                       'rate': 16000.0,
                       'bits_per_sample': 32}

        y = self.chain.apply(x, src_info=src_info, target_info=target_info)

        # sox might misbehave sometimes by giving nan/inf if sequences are too short (or silent)
        # and the effect chain includes eg `pitch`
        if torch.isnan(y).any() or torch.isinf(y).any():
            print("ERROR!")
            return x
        return y.squeeze(0)


class AudioAugmentDataset(BaseWrapperDataset):
    def __init__(
            self, dataset, normalize, split, augmentations,
            reverb_strength, reverb_damping, reverb_room_std, 
            pitch_shift_std, speed_std, snr_min, snr_max,
            augment_source_prob, augment_target_prob,
            match_source_target_aug,
    ):

        print("AUGMENT DATASET")
        super().__init__(dataset)
        # assert args.augment_source_prob > 0 or args.augment_target_prob > 0, \
        #     "Atleast one of source and target needs to be augmented"
        self.normalize = normalize
        self.split = split
        self.augmentations = augmentations
        self.reverb_strength = reverb_strength
        self.reverb_damping = reverb_damping
        self.reverb_room_std = reverb_room_std
        self.pitch_shift_std = pitch_shift_std
        self.speed_std = speed_std
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.augment_source_prob = augment_source_prob
        self.augment_target_prob = augment_target_prob
        self.match_source_target_aug = match_source_target_aug

        self.noise_root = Path("/checkpoint/anuroops/data/musan")
        self.noise_files = list(self.noise_root.rglob("*.wav"))

        effect_chain = augment.EffectChain().rate("-q", 16_000)
        augmentations = augmentations.split(",")
        for aug in sorted(augmentations):
            if aug == "pitch":
                effect_chain = effect_chain.pitch("-q", self.random_pitch_shift)
            elif aug == "speed":
                # effect_chain = effect_chain.speed(self.random_speed)
                pass
            elif aug == "reverb":
                effect_chain = effect_chain.reverb(reverb_strength, reverb_damping,
                                                   self.random_room_size).channels()
            elif aug == "additive":
                pass
            else:
                raise NotImplementedError(f"Unknown augmentation: {aug}")
        self.runner = ChainRunner(effect_chain)

    def random_pitch_shift(self):
        return np.random.randn() * self.pitch_shift_std

    def random_speed(self):
        rand = min(max(-3, np.random.randn()), 3)
        return max(1. + rand * self.speed_std, 0.1)

    def random_room_size(self):
        return min(np.abs(np.random.randn() * self.reverb_room_std), 100.)

    def random_noise(self, numframes):
        import soundfile as sf

        def _noise_generator():
            wav, curr_sample_rate = sf.read(np.random.choice(self.noise_files))
            if wav.shape[0] > numframes:
                start = np.random.randint(0, wav.shape[0] - numframes)
                wav = wav[start: start + numframes]
            else:
                wav = np.repeat(wav, repeats=numframes // wav.shape[0] + 1)
                wav = wav[:numframes]
            return torch.tensor(wav).float()
        return _noise_generator

    def __getitem__(self, index):
        def _maybe_normalize(feats):
            if not self.normalize:
                return feats
            with torch.no_grad():
                feats = F.layer_norm(feats, feats.shape)
                return feats / feats.abs().max()

        def _maybe_aug(feats, prob):
            if self.split == "train" and np.random.random() < prob:
                aug = self.runner(feats)
                if "additive" in self.augmentations:
                    noise_generator = self.random_noise(feats.shape[0])
                    snr = np.random.random() * (self.snr_max - self.snr_min) + self.snr_min
                    aug = augment.EffectChain().additive_noise(noise_generator, snr=snr) \
                        .apply(aug, src_info={'rate': 16000}, target_info={'rate': 16000})
                if "speed" in self.augmentations:
                    x = aug.numpy()
                    y = librosa.resample(x, self.random_speed(), 1)
                    diff = abs(len(x) - len(y))
                    if len(y) > len(x):
                        # Truncate noise
                        y = y[diff // 2 : -((diff + 1) // 2)]
                    elif len(y) < len(x):
                        # Assume the time-axis is the first: (Time, Channel)
                        pad_width = [(diff // 2, (diff + 1) // 2)] + [
                            (0, 0) for _ in range(y.ndim - 1)
                        ]
                        y = np.pad(
                            y, pad_width=pad_width, constant_values=0, mode="constant"
                        )
                    aug = aug.new_tensor(y)
            else:
                aug = feats
            return _maybe_normalize(aug)

        item = self.dataset[index]
        source = item["source"]
        item["original_source"] = source
        item["source"] = _maybe_aug(source.clone(), self.augment_source_prob)
        if self.match_source_target_aug:
            item["target"] = source.clone()
        else:
            item["target"] = _maybe_aug(source.clone(), self.augment_target_prob)
        # print(
        #     "Source:", source.shape, source.abs().sum(),
        #     (source - item["source"]).abs().sum(), 
        #     (source - item["target"]).abs().sum(),
        #     (item["source"] - item["target"]).abs().sum(),
        # )
        return item
