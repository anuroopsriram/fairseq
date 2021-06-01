from augment.effects import EffectChain
import numpy as np
from pathlib import Path
import soundfile as sf

import torch
from fairseq.data.base_wrapper_dataset import BaseWrapperDataset
from torch.nn import functional as F


class ChainRunner:
    def __init__(self, chain):
        self.chain = chain

    def __call__(self, x):
        """
        x: torch.Tensor, (channels, length). Must be placed on CPU.
        """
        x = x.view(1, -1)
        src_info = {
            'channels': x.size(0),  # number of channels
            # 'length': x.size(1),  # length of the sequence
            'precision': 32,  # precision (16, 32 bits)
            'rate': 16000.0,  # sampling rate
            'bits_per_sample': 32
        }

        target_info = {
            'channels': 1,
            'length': x.size(1),
            'precision': 32,
            'rate': 16000.0,
            'bits_per_sample': 32
        }

        y = self.chain.apply(x, src_info=src_info, target_info=target_info)

        # sox might misbehave sometimes by giving nan/inf if sequences are too short (or silent)
        # and the effect chain includes eg `pitch`
        if torch.isnan(y).any() or torch.isinf(y).any():
            print("ERROR!")
            return x

        aug = torch.zeros_like(x)
        if y.size(1) > x.size(1):
            aug = y[:, :x.size(1)]
        else:
            aug[:, :y.size(1)] = y
        return aug.squeeze(0)


class AdditiveNoise:
    def __init__(self, snr_min: float, snr_max: float, prob: float, noise_root: str="/checkpoint/anuroops/data/musan"):
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.prob = prob
        self.noise_root = Path(noise_root)
        self.noise_files = list(self.noise_root.rglob("*.wav"))

    def __call__(self, x, src_info, dst_info):
        if np.random.rand() > self.prob:
            return x, src_info, dst_info
        else:
            numframes = x.shape[1]
            noise_instance, _ = sf.read(np.random.choice(self.noise_files))
            if noise_instance.shape[0] > numframes:
                start = np.random.randint(0, noise_instance.shape[0] - numframes)
                noise_instance = noise_instance[start: start + numframes]
            else:
                noise_instance = np.repeat(noise_instance, repeats=numframes // noise_instance.shape[0] + 1)
                noise_instance = noise_instance[:numframes]
            noise_instance = x.new_tensor(noise_instance)

            snr = np.random.random() * (self.snr_max - self.snr_min) + self.snr_min
            r = np.exp(snr * np.log(10) / 10)
            self.coeff = r / (1.0 + r)
            noised = self.coeff * x + (1.0 - self.coeff) * noise_instance.view_as(x)
            return noised, src_info, dst_info


class AugmentationEffectChain(EffectChain):
    def additive_noise(self, snr_min: float, snr_max: float, prob: float):
        self._chain.append(AdditiveNoise(snr_min, snr_max, prob))
        return self


# def random_pitch(prob, pitch_shift_std):
#     def _func():
#         if np.random.rand() > prob:
#             pitch_shift = 0
#         else:
#             pitch_shift = np.clip(np.random.randn(), -2, 2) * pitch_shift_std
#         return pitch_shift
#     return _func


class AudioAugmentData(BaseWrapperDataset):
    def __init__(
            self, dataset, normalize, split, augmentations,
            augment_source_prob, augment_target_prob,
            match_source_target_aug, sample_rate,
            pitch_shift_std=0., speed_std=0., reverb_strength_std=0.,
            reverb_damping=0., reverb_room_size=0.,
            snr_min=0., snr_max=0.
    ):
        super().__init__(dataset)
        self.normalize = normalize
        self.split = split
        self.augmentations = sorted(augmentations.split(","))
        self.augment_source_prob = augment_source_prob
        self.augment_target_prob = augment_target_prob
        self.match_source_target_aug = match_source_target_aug
        self.sample_rate = sample_rate

        self.pitch_shift_std = pitch_shift_std
        self.speed_std = speed_std
        self.reverb_strength_std = reverb_strength_std
        self.reverb_damping = reverb_damping
        self.reverb_room_size = reverb_room_size
        self.snr_min = snr_min
        self.snr_max = snr_max

        self.source_runner = self.create_effect_chain(self.augment_source_prob)
        self.target_runner = self.create_effect_chain(self.augment_target_prob)

    def create_effect_chain(self, prob):
        def _random_pitch():
            if np.random.rand() > prob:
                pitch_shift = 0
            else:
                pitch_shift = np.clip(np.random.randn(), -2, 2) * self.pitch_shift_std
            # print("Pitch", pitch_shift)
            return pitch_shift

        def _random_speed():
            if np.random.rand() > prob:
                speed_perturb = 1
            else:
                speed_perturb = 1. + np.clip(np.random.randn(), -2, 2) * self.speed_std
            # print("Speed", speed_perturb)
            return speed_perturb
        
        def _random_reverb():
            if np.random.rand() > prob:
                reverb_strength = 0
            else:
                reverb_strength = np.abs(np.random.randn() * self.reverb_strength_std)
            # print("Reverb", reverb_strength)
            return np.clip(reverb_strength, 0, 100)

        effect_chain = AugmentationEffectChain()
        if "pitch" in self.augmentations:
            effect_chain = effect_chain.pitch(_random_pitch).rate(self.sample_rate)
            # effect_chain = effect_chain.pitch(random_pitch(prob, self.pitch_shift_std)).rate(self.sample_rate)
        if "speed" in self.augmentations:
            effect_chain = effect_chain.speed(_random_speed).rate(self.sample_rate)
        if "reverb" in self.augmentations:
            effect_chain = effect_chain.reverb(_random_reverb, self.reverb_damping, self.reverb_room_size).channels().rate(self.sample_rate)
        if "additive" in self.augmentations:
            effect_chain.additive_noise(self.snr_min, self.snr_max, prob).rate(self.sample_rate)
        effect_chain = effect_chain.rate(self.sample_rate)
        return ChainRunner(effect_chain)

    def __getitem__(self, index):
        def _normalize(feats):
            if self.normalize:
                feats = F.layer_norm(feats, feats.shape)
                feats = feats / feats.abs().max()
            return feats

        item = self.dataset[index]
        source = item["source"]
        item["original_source"] = source
        item["source"] = self.source_runner(source.clone())
        if self.match_source_target_aug:
            item["target"] = source.clone()
        else:
            item["target"] = self.target_runner(source.clone())
        item["source"] = _normalize(item["source"])
        item["target"] = _normalize(item["target"])
        assert item["source"].shape == item["original_source"].shape, f'{item["source"].shape} vs {item["original_source"].shape}' 
        assert item["target"].shape == item["original_source"].shape, f'{item["target"].shape} vs {item["original_source"].shape}'
        return item
