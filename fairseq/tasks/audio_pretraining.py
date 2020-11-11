# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import sys

from fairseq.data.data_utils import post_process

from fairseq.data import FileAudioDataset, Dictionary, AddTargetDataset, encoders
from . import FairseqTask, register_task
from .. import utils
from ..data.audio.audio_augment_dataset import AudioAugmentDataset
from ..data.audio.raw_audio_dataset import LogMelAudioDataset
from ..logging import metrics


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


@register_task("audio_pretraining")
class AudioPretrainingTask(FairseqTask):
    """

    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="path to data directory")
        parser.add_argument(
            "--sample-rate",
            default=16000,
            type=int,
            help="target sample rate. audio files will be up/down sampled to this rate",
        )
        parser.add_argument(
            "--normalize",
            action="store_true",
            help="if set, normalizes input to have 0 mean and unit variance",
        )
        parser.add_argument(
            "--max-sample-size",
            default=None,
            type=int,
            help="max sample size to crop to for batching. default = min sample length",
        )
        parser.add_argument(
            "--min-sample-size",
            default=None,
            type=int,
            help="min sample size to crop to for batching. default = same as --max-sample-size",
        )

        parser.add_argument(
            "--enable-padding",
            action="store_true",
            help="pad shorter samples instead of cropping",
        )

        parser.add_argument(
            "--labels",
            type=str,
            default=None,
            help="extension of the label file to load, if any",
        )

        parser.add_argument(
            "--logmel",
            action="store_true",
            help="if set, creates logmel features",
        )
        parser.add_argument(
            "--num-mel-bins",
            default=80,
            type=int,
            help="number of mel filter banks",
        )
        parser.add_argument(
            "--frame-length",
            default=25,
            type=float,
            help="Frame length to compute MEL filter banks",
        )
        parser.add_argument(
            "--frame-shift",
            default=10,
            type=float,
            help="Frame length to compute MEL filter banks",
        )
        parser.add_argument(
            "--specaug-prob",
            default=0.,
            type=float,
            help="prob of applying specaug",
        )
        parser.add_argument(
            "--eval-wer",
            action="store_true",
            help="compute WER",
        )
        parser.add_argument(
            "--eval-wer-remove-bpe",
            "--eval-wer-post-process",
            default="letter",
            help="remove BPE tokens before scoring (can be set to sentencepiece, letter, and more)",
        )
        parser.add_argument(
            "--augment-audio",
            action='store_true',
            help="apply data augmentation",
        )
        parser.add_argument(
            "--augment-source-prob",
            default=0.,
            type=float,
            help="probabilty of augmenting source audio",
        )
        parser.add_argument(
            "--augment-target-prob",
            default=0.,
            type=float,
            help="probabilty of augmenting target audio",
        )
        parser.add_argument(
            "--augmentations",
            default="additive,pitch,speed,reverb",
            type=str,
            help="max pitch shift",
        )
        parser.add_argument(
            "--snr-min",
            default=5.,
            type=float,
        )
        parser.add_argument(
            "--snr-max",
            default=15.,
            type=float,
        )
        parser.add_argument(
            "--pitch-shift-std",
            default=200.,
            type=float,
        )
        parser.add_argument(
            "--speed-std",
            default=0.1,
            type=float,
        )

        # parser.add_argument(
        #     "--augment-pitch-shift",
        #     default=300.,
        #     type=float,
        #     help="max pitch shift",
        # )
        # parser.add_argument(
        #     "--augment-time-drop",
        #     default=40,
        #     type=int,
        #     help="max time drop in number of frames",
        # )

    def __init__(self, args, source_dictionary=None):
        super().__init__(args)
        self._target_dictionary = None
        self._source_dictionary = source_dictionary
        self.is_ctc = args.criterion == "ctc"

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        return cls(args)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        manifest = os.path.join(self.args.data, "{}.tsv".format(split))
        if hasattr(self.args, 'logmel') and self.args.logmel:
            self.datasets[split] = LogMelAudioDataset(
                manifest,
                sample_rate=self.args.sample_rate,
                max_sample_size=self.args.max_sample_size,
                min_sample_size=self.args.max_sample_size,
                min_length=self.args.min_sample_size,
                pad=self.args.labels is not None or self.args.enable_padding,
                num_mel_bins=self.args.num_mel_bins,
                frame_length=self.args.frame_length,
                frame_shift=self.args.frame_shift,
                specaug_prob=(self.args.specaug_prob if split == 'train' else 0.),
            )
        else:
            self.datasets[split] = FileAudioDataset(
                manifest,
                sample_rate=self.args.sample_rate,
                max_sample_size=self.args.max_sample_size,
                min_sample_size=self.args.max_sample_size,
                min_length=self.args.min_sample_size,
                pad=self.args.labels is not None or self.args.enable_padding,
                normalize=(self.args.normalize and not self.args.augment_audio),
            )

        if self.args.labels:
            dict_path = os.path.join(self.args.data, f"dict.{self.args.labels}.txt")
            self._target_dictionary = Dictionary.load(dict_path)
            label_path = os.path.join(self.args.data, f"{split}.{self.args.labels}")
            labels = []
            with open(label_path, "r") as f:
                for line in f:
                    labels.append(line)

            process_label = LabelEncoder(self.target_dictionary)

            self.datasets[split] = AddTargetDataset(
                self.datasets[split],
                labels,
                pad=self.target_dictionary.pad(),
                eos=self.target_dictionary.eos(),
                batch_targets=True,
                process_label=process_label,
                add_to_input=not self.is_ctc,
            )
        if self.args.augment_audio:
            self.datasets[split] = AudioAugmentDataset(self.datasets[split], self.args, self.args.normalize)

    @property
    def source_dictionary(self):
        return self._source_dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self._target_dictionary

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return (sys.maxsize, sys.maxsize)

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if getattr(self.args, "eval_wer", False) and not self.is_ctc:
            metrics = self._inference_with_wer(self.sequence_generator, sample, model)
            logging_output["_num_word_errors"] = metrics["num_word_errors"]
            logging_output["_num_words"] = metrics["num_words"]
        return loss, sample_size, logging_output

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, 'eval_wer', False) and not self.is_ctc:
            self.sequence_generator = self.build_generator([model], args)
            self.tokenizer = encoders.build_tokenizer(args)
        return model

    def _inference_with_wer(self, generator, sample, model):
        import editdistance

        def decode(toks, escape_unk=True):
            s = self.target_dictionary.string(
                toks.int().cpu(),
                self.args.eval_wer_remove_bpe,
                escape_unk=escape_unk,
                extra_symbols_to_ignore={generator.eos},
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        num_errors = 0
        num_tokens = 0
        gen_out = self.inference_step(generator, [model], sample, None)
        for i in range(len(gen_out)):
            hyp = decode(gen_out[i][0]["tokens"])
            ref = decode(
                utils.strip_pad(sample["target"][i], self.target_dictionary.pad()),
                escape_unk=True,
            )
            hyp_words = hyp.split()
            ref_words = ref.split()
            num_errors += editdistance.eval(hyp_words, ref_words)
            num_tokens += len(ref_words)
        return {
            "num_word_errors": num_errors,
            "num_words": num_tokens,
        }

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        num_word_errors = sum(log.get("_num_word_errors", 0) for log in logging_outputs)
        metrics.log_scalar("_num_word_errors", num_word_errors)
        num_words = sum(log.get("_num_words", 0) for log in logging_outputs)
        metrics.log_scalar("_num_words", num_words)
        if num_words > 0:
            metrics.log_derived(
                "wer",
                lambda meters: round(meters["_num_word_errors"].sum * 100.0 / meters["_num_words"].sum, 3)
                if meters["_num_words"].sum > 0 else float("nan")
            )
