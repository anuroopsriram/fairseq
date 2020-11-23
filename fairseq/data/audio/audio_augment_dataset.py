# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
            return x
        return y.squeeze(0)


class AudioAugmentDataset(BaseWrapperDataset):
    def __init__(self, dataset, args, normalize, split):
        print("AUGMENT DATASET")
        super().__init__(dataset)
        # assert args.augment_source_prob > 0 or args.augment_target_prob > 0, \
        #     "Atleast one of source and target needs to be augmented"
        self.normalize = normalize
        self.args = args
        self.split = split
        self.noise_root = Path("/checkpoint/anuroops/data/musan")
        self.noise_files = list(self.noise_root.rglob("*.wav"))

        effect_chain = augment.EffectChain().rate("-q", 16_000)
        augmentations = self.args.augmentations.split(",")
        for aug in sorted(augmentations):
            if aug == "pitch":
                effect_chain = effect_chain.pitch("-q", self.random_pitch_shift)
            elif aug == "speed":
                effect_chain = effect_chain.speed(self.random_speed)
            elif aug == "reverb":
                effect_chain = effect_chain.reverb(50, 50, self.random_room_size).channels()
            elif aug == "additive":
                pass
            else:
                raise NotImplementedError(f"Unknown augmentation: {aug}")
        self.runner = ChainRunner(effect_chain)

    def random_pitch_shift(self):
        return np.random.randn() * self.args.pitch_shift_std

    def random_speed(self):
        return 1. + np.random.randn() * self.args.speed_std

    def random_room_size(self):
        return np.random.randint(0, 100)

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
                if "additive" in self.args.augmentations:
                    noise_generator = self.random_noise(feats.shape[0])
                    snr = np.random.random() * (self.args.snr_max - self.args.snr_min) + self.args.snr_min
                    aug = augment.EffectChain().additive_noise(noise_generator, snr=snr) \
                        .apply(aug, src_info={'rate': 16000}, target_info={'rate': 16000})
            else:
                aug = feats
            return _maybe_normalize(aug)

        item = self.dataset[index]
        source = item["source"]
        item["original_source"] = source
        item["source"] = _maybe_aug(source.clone(), self.args.augment_source_prob)
        item["target"] = _maybe_aug(source.clone(), self.args.augment_target_prob)
        return item


# class AudioAugmentDataset(BaseWrapperDataset):
#     def __init__(self, dataset, args, normalize):
#         super().__init__(dataset)
#         assert args.augment_source_prob > 0 or args.augment_target_prob > 0, \
#             "Atleast one of source and target needs to be augmented"
#         self.normalize = normalize
#         self.args = args
#         # self.augment_source_prob = args.augment_source_prob
#         # self.augment_target_prob = args.augment_target_prob
#         # self.pitch_shift = args.augment_pitch_shift
#         self.noise_root = Path("/checkpoint/anuroops/data/musan")
#         self.noise_files = list(self.noise_root.rglob("*.wav"))
#         self.snr_min = 5
#         self.snr_max = 10
#
#         effect_chain = augment.EffectChain() \
#             .pitch("-q", self.random_pitch_shift) \
#             .rate("-q", 16_000) \
#             .speed(self.random_speed) \
#             .reverb(50, 50, self.random_room_size).channels()  # \
#             # .time_dropout(max_frames=args.augment_time_drop)
#         self.runner = ChainRunner(effect_chain)
#
#     def random_pitch_shift(self):
#         # return np.random.randint(-self.pitch_shift, self.pitch_shift)
#         return np.random.randn() * self.args.pitch_shift_std
#
#     def random_room_size(self):
#         # return np.random.randint(0, 100)
#         return np.random.randn() * self.args.room_size_std
#
#     def random_speed(self):
#         a, b = 0.8, 1.2
#         return np.random.random() * (b - a) + a
#
#     def random_noise(self, numframes):
#         import soundfile as sf
#
#         def _noise_generator():
#             wav, curr_sample_rate = sf.read(np.random.choice(self.noise_files))
#             if wav.shape[0] > numframes:
#                 start = np.random.randint(0, wav.shape[0] - numframes)
#                 wav = wav[start: start + numframes]
#                 # print("SHAPES2", wav.shape, numframes)
#             else:
#                 wav = np.repeat(wav, repeats=numframes // wav.shape[0] + 1)
#                 wav = wav[:numframes]
#             return torch.tensor(wav).float()
#         return _noise_generator
#
#     def __getitem__(self, index):
#         def _maybe_normalize(feats):
#             if not self.normalize:
#                 return feats
#             with torch.no_grad():
#                 return F.layer_norm(feats, feats.shape)
#
#         item = self.dataset[index]
#         source = item["source"]
#         item["original_source"] = source
#
#         noise_generator = self.random_noise(source.shape[0])
#
#         if np.random.random() < self.augment_source_prob:
#             source_aug = self.runner(source)
#             snr = np.random.random() * (self.snr_max - self.snr_min) + self.snr_min
#             source_aug = augment.EffectChain().additive_noise(noise_generator, snr=snr)\
#                 .apply(source_aug, src_info={'rate': 16000}, target_info={'rate': 16000})
#             item["source"] = _maybe_normalize(source_aug)
#         else:
#             item["source"] = _maybe_normalize(source)
#
#         if np.random.random() < self.augment_target_prob:
#             target_aug = self.runner(source)
#             target_aug = augment.EffectChain().additive_noise(noise_generator, snr=15) \
#                 .apply(target_aug, src_info={'rate': 16000}, target_info={'rate': 16000})
#             item["target"] = _maybe_normalize(target_aug)
#         else:
#             item["target"] = _maybe_normalize(source)
#         return item
