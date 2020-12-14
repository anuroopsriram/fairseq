# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import logging
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple

from torch.nn import BatchNorm1d

from fairseq.modules.relative_positional_attention import RelativePositionalMultiHeadAttention

from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.models import BaseFairseqModel, register_model, register_model_architecture
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
    MultiheadAttention,
    SamePad,
    TransposeLast,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import buffered_arange


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

    def __repr__(self):
        return f"Lambda({self.func})"


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class MaybeBatchNorm(nn.Module):
    def __init__(self, num_channels, enabled=True):
        super().__init__()
        self.bn = None
        if enabled:
            self.bn = nn.Sequential(
                Permute(0, 2, 1),  # (B, T, D) --> (B, D, T)
                BatchNorm1d(num_channels),
                Permute(0, 2, 1),  # (B, D, T) --> (B, T, D)
            )

    def forward(self, x):
        if self.bn is not None:
            return self.bn(x)
        return x


@register_model("wav2vec2")
class Wav2Vec2Model(BaseFairseqModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""

        parser.add_argument(
            "--extractor-mode",
            choices=["default", "layer_norm"],
            help="mode for feature extractor. default has a single group norm with d groups in the first conv block, whereas layer_norm has layer norms in every block (meant to use with --normalize)",
        )

        parser.add_argument(
            "--encoder-layers",
            type=int,
            metavar="L",
            help="num encoder layers in the transformer",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )

        parser.add_argument(
            "--dropout",
            type=float,
            metavar="D",
            help="dropout probability for the transformer",
        )

        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )

        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )

        parser.add_argument(
            "--final-dim",
            type=int,
            metavar="D",
            help="project final representations and targets to this many dimensions",
        )

        parser.add_argument(
            "--layer-norm-first",
            action="store_true",
            help="apply layernorm first in the transformer",
        )

        parser.add_argument(
            "--encoder-layerdrop",
            type=float,
            help="probability of dropping a tarnsformer layer",
        )

        parser.add_argument(
            "--conv-feature-layers",
            type=str,
            metavar="EXPR",
            help="convolutional feature extraction layers [(dim, kernel_size, stride), ...]",
        )

        parser.add_argument(
            "--logit-temp", type=float, help="temperature to divide logits by"
        )

        parser.add_argument(
            "--quantize-targets", action="store_true", help="use quantized targets"
        )

        parser.add_argument(
            "--quantize-input", action="store_true", help="use quantized inputs"
        )

        parser.add_argument(
            "--feature-grad-mult",
            type=float,
            help="multiply feature extractor var grads by this",
        )

        parser.add_argument(
            "--latent-vars",
            type=int,
            metavar="N",
            help="number of latent variables V in each group of the codebook",
        )

        parser.add_argument(
            "--latent-groups",
            type=int,
            metavar="N",
            help="number of groups G of latent variables in the codebook",
        )

        parser.add_argument(
            "--latent-dim",
            type=int,
            metavar="N",
            help="if set, uses this dimensionality for latent variables. otherwise uses final_dim / latent_groups",
        )

        parser.add_argument("--mask-length", type=int, help="mask length")

        parser.add_argument(
            "--mask-prob", type=float, help="probability of replacing a token with mask"
        )

        parser.add_argument(
            "--mask-selection",
            type=str,
            choices=["static", "uniform", "normal", "poisson"],
            help="how to choose masks",
        )

        parser.add_argument(
            "--mask-other",
            type=float,
            help="secondary mask argument (used for more complex distributions), see help in compute_mask_indices",
        )

        parser.add_argument(
            "--no-mask-overlap",
            action="store_true",
            help="whether to allow masks to overlap",
        )

        parser.add_argument(
            "--mask-min-space",
            type=int,
            help="min space between spans (if no overlap is enabled)",
        )

        parser.add_argument(
            "--mask-channel-length",
            type=int,
            help="repeat the mask indices multiple times",
        )

        parser.add_argument(
            "--mask-channel-prob",
            type=float,
            help="probability of replacing a token with mask",
        )

        parser.add_argument(
            "--mask-channel-selection",
            type=str,
            choices=["static", "uniform", "normal", "poisson"],
            help="how to choose masks",
        )

        parser.add_argument(
            "--mask-channel-other",
            type=float,
            help="secondary mask argument (used for more complex distributions), see help in compute_mask_indices",
        )

        parser.add_argument(
            "--no-mask-channel-overlap",
            action="store_true",
            help="whether to allow masks to overlap",
        )

        parser.add_argument(
            "--mask-channel-min-space",
            type=int,
            help="min space between spans (if no overlap is enabled)",
        )

        parser.add_argument(
            "--dropout-input",
            type=float,
            metavar="D",
            help="dropout to apply to the input (after feat extr)",
        )

        parser.add_argument(
            "--dropout-features",
            type=float,
            metavar="D",
            help="dropout to apply to the features (after feat extr)",
        )

        parser.add_argument(
            "--num-negatives", type=int, metavar="N", help="number of negative examples"
        )

        parser.add_argument(
            "--negatives-from-everywhere",
            action="store_true",
            help="sample negatives from everywhere, not just masked states",
        )

        parser.add_argument(
            "--cross-sample-negatives",
            type=int,
            metavar="N",
            help="num of cross sampled negatives",
        )

        parser.add_argument(
            "--codebook-negatives",
            type=int,
            metavar="N",
            help="num of codebook sampled negatives",
        )

        parser.add_argument(
            "--conv-pos",
            type=int,
            metavar="N",
            help="number of filters for convolutional positional embeddings",
        )

        parser.add_argument(
            "--conv-pos-groups",
            type=int,
            metavar="N",
            help="number of groups for convolutional positional embedding",
        )

        parser.add_argument(
            "--latent-temp",
            type=str,
            metavar="D",
            help="temperature for latent variable sampling. can be tuple of 3 values (start, end, decay)",
        )

        parser.add_argument(
            "--target-glu", action="store_true", help="adds projection + glu to targets"
        )

        parser.add_argument(
            "--conv-bias", action="store_true", help="include bias in conv encoder"
        )

        parser.add_argument(
            "--in-d", type=int, default=1, help="number of input channels"
        )

        # Conformer Arguments
        parser.add_argument(
            "--transformer-type",
            choices=('transformer', 'conformer'),
            help="whether to use conformer or transformer layers",
            default="transformer"
        )
        parser.add_argument(
            "--activation-fn1",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
            default="swish"
        )
        parser.add_argument(
            "--activation-fn2",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
            default="gelu"
        )
        parser.add_argument(
            "--conformer-kernel-size",
            type=int,
            help="convolution kernel size in conformer",
            default=32,
        )
        parser.add_argument(
            "--ffn-scale",
            type=float,
            help="ffn scale",
            default=0.5
        )
        parser.add_argument(
            "--no-expand-ffn",
            default=False,
            action='store_true',
            help="Conformer FFN expansion",
        )
        parser.add_argument(
            '--use-rel-posn-mha',
            default=False,
            action='store_true'
        )
        parser.add_argument(
            '--rel-posn-mha-list',
            default=None,
            metavar="EXPR",
            type=str,
        )
        parser.add_argument(
            '--num-relpos-embeds',
            default=768,
            type=int
        )
        parser.add_argument(
            '--lin-dropout',
            default=0.1,
            type=float
        )
        parser.add_argument(
            '--conformer-list',
            default=None,
            metavar="EXPR",
            type=str,
        )
        parser.add_argument(
            '--conformer-mha-list',
            default=None,
            metavar="EXPR",
            type=str,
        )
        parser.add_argument(
            "--projection-mlp-context", action="store_true", help="adds projection MLP a la BYOL"
        )
        parser.add_argument(
            "--target-mlp-context", action="store_true",
            help="adds projection MLP a la BYOL to target, before quantization"
        )
        parser.add_argument("--mlp-nobn", action="store_true", default=False)
        parser.add_argument("--mlp-scale", type=int, default=2)
        parser.add_argument(
            "--mlp-activation",
            choices=utils.get_available_activation_fns(),
            default="relu"
        )
        parser.add_argument(
            '--apply-encoder-to-target',
            default=False,
            action='store_true'
        )
        parser.add_argument(
            '--mask-target',
            default=False,
            action='store_true'
        )

    def __init__(self, args):
        super().__init__()
        self.args = args

        feature_enc_layers = eval(args.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=args.extractor_mode,
            conv_bias=args.conv_bias,
            in_d=args.in_d,
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, args.encoder_embed_dim)
            if self.embed != args.encoder_embed_dim and not args.quantize_input
            else None
        )

        self.mask_prob = args.mask_prob
        self.mask_selection = args.mask_selection
        self.mask_other = args.mask_other
        self.mask_length = args.mask_length
        self.no_mask_overlap = args.no_mask_overlap
        self.mask_min_space = args.mask_min_space

        self.mask_channel_prob = args.mask_channel_prob
        self.mask_channel_selection = args.mask_channel_selection
        self.mask_channel_other = args.mask_channel_other
        self.mask_channel_length = args.mask_channel_length
        self.no_mask_channel_overlap = args.no_mask_channel_overlap
        self.mask_channel_min_space = args.mask_channel_min_space

        self.dropout_input = nn.Dropout(args.dropout_input)
        self.dropout_features = nn.Dropout(args.dropout_features)

        self.feature_grad_mult = args.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = args.num_negatives
        self.cross_sample_negatives = args.cross_sample_negatives
        self.codebook_negatives = args.codebook_negatives
        self.negatives_from_everywhere = args.negatives_from_everywhere

        self.logit_temp = args.logit_temp

        if args.quantize_input:
            vq_dim = args.latent_dim if args.latent_dim > 0 else args.encoder_embed_dim
            self.input_quantizer = (
                GumbelVectorQuantizer(
                    dim=args.encoder_embed_dim,
                    num_vars=args.latent_vars,
                    temp=eval(args.latent_temp),
                    groups=args.latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                )
                if not args.same_quantizer
                else self.quantizer
            )
            self.project_inp = nn.Linear(vq_dim, args.encoder_embed_dim)

        final_dim = args.final_dim if args.final_dim > 0 else args.encoder_embed_dim

        if args.quantize_targets:
            vq_dim = args.latent_dim if args.latent_dim > 0 else final_dim
            self.quantizer = GumbelVectorQuantizer(
                dim=self.embed if not self.args.apply_encoder_to_target else args.encoder_embed_dim,
                num_vars=args.latent_vars,
                temp=eval(args.latent_temp),
                groups=args.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
            )
            # self.project_q = nn.Linear(vq_dim, final_dim)
            self.project_q = self._create_project_q(args, vq_dim, final_dim)
        else:
            # self.project_q = nn.Linear(self.embed, final_dim)
            self.project_q = self._create_project_q(args, self.embed, final_dim)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(args.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(args)
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if args.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        if hasattr(args, "projection_mlp_context") and args.projection_mlp_context:
            self.final_proj = self._create_mlp(
                args.encoder_embed_dim,
                args.encoder_embed_dim * args.mlp_scale,
                final_dim,
                not args.mlp_nobn,
                args.mlp_activation
            )
            # self.final_proj = nn.Sequential(
            #     nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim * 2),
            #     Permute(0, 2, 1),  # (B, T, D) --> (B, D, T)
            #     nn.BatchNorm1d(args.encoder_embed_dim * 2),
            #     Permute(0, 2, 1),  # (B, D, T) --> (B, T, D)
            #     nn.ReLU(),
            #     nn.Linear(args.encoder_embed_dim * 2, final_dim)
            # )
        else:
            self.final_proj = nn.Linear(args.encoder_embed_dim, final_dim)

    def _create_mlp(self, in_dim, inner_dim, out_dim, use_bn, act):
        activation_fn = utils.get_activation_fn(act)
        return nn.Sequential(
            nn.Linear(in_dim, inner_dim),
            MaybeBatchNorm(inner_dim, use_bn),
            Lambda(activation_fn),
            nn.Linear(inner_dim, out_dim)
        )

    def _create_project_q(self, args, in_dim, out_dim):
        if hasattr(args, "target_mlp_context") and args.target_mlp_context:
            return self._create_mlp(
                in_dim, in_dim * args.mlp_scale, out_dim,
                not args.mlp_nobn,
                args.mlp_activation
            )
            # return nn.Sequential(
            #     nn.Linear(in_dim, in_dim * 2),
            #     Permute(0, 2, 1),  # (B, T, D) --> (B, D, T)
            #     nn.BatchNorm1d(in_dim * 2),
            #     Permute(0, 2, 1),  # (B, D, T) --> (B, T, D)
            #     nn.ReLU(),
            #     nn.Linear(in_dim * 2, out_dim)
            # )
        else:
            return nn.Linear(in_dim, out_dim)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    @classmethod
    def build_model(cls, args, task=None):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        return cls(args)

    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def sample_negatives(self, y, num):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        cross_high = tsz * bsz
        high = tsz
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def compute_preds(self, x, y, negatives):

        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0)

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)

        logits /= self.logit_temp

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        return logits

    def compute_penalty(self, pen1, pen2):
        return (pen1 + pen2) / 2

    def forward(self, source, padding_mask=None, mask=True, features_only=False, target=None):
        features, features_pen = self.compute_features(source, self.feature_grad_mult, for_target=False)
        if target is None:
            unmasked_features = features.clone()
        else:
            unmasked_features, features_pen2 = self.compute_features(target, self.feature_grad_mult, for_target=True)
            features_pen = self.compute_penalty(features_pen, features_pen2)

        if padding_mask is not None:
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask.all(-1)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
            if self.args.apply_encoder_to_target:
                unmasked_features = self.post_extract_proj(unmasked_features)

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
            if mask_indices is not None:
                y = unmasked_features[mask_indices].view(unmasked_features.size(0), -1, unmasked_features.size(-1))
            else:
                y = unmasked_features
        else:
            x = features
            y = unmasked_features
            mask_indices = None

        x = self.encoder(x, padding_mask=padding_mask)

        if features_only:
            return {"x": x, "padding_mask": padding_mask}

        if self.args.apply_encoder_to_target:
            y = self.apply_encoder_to_target(padding_mask, y)

        if self.quantizer:
            q = self.quantizer(y, produce_targets=False)
            y = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]

            y = self.project_q(y)

            if self.negatives_from_everywhere:
                neg_cands, *_ = self.quantizer(unmasked_features, produce_targets=False)
                negs, _ = self.sample_negatives(neg_cands, y.size(1))
                negs = self.project_q(negs)

            else:
                negs, _ = self.sample_negatives(y, y.size(1))

            if self.codebook_negatives > 0:
                cb_negs = self.quantizer.sample_from_codebook(
                    y.size(0) * y.size(1), self.codebook_negatives
                )
                cb_negs = cb_negs.view(
                    self.codebook_negatives, y.size(0), y.size(1), -1
                )  # order doesnt matter
                cb_negs = self.project_q(cb_negs)
                negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(unmasked_features, y.size(1))
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(y, y.size(1))

        x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)

        result = {"x": x, "padding_mask": padding_mask, "features_pen": features_pen}

        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp

        return result

    def apply_encoder_to_target(self, padding_mask, y):
        with torch.no_grad():
            if self.args.mask_target:
                y = self.encoder(y, padding_mask=padding_mask)
            else:
                y = self.encoder(y)
        return y

    def compute_features(self, source, feature_grad_mult, for_target=False):
        # feature_grad_mult = self.feature_grad_mult
        if feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        features_pen = features.float().pow(2).mean()
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        return features, features_pen

    def quantize(self, x):
        assert self.quantizer is not None
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return self.quantizer.forward_idx(x)

    def extract_features(self, source, padding_mask, mask=False):
        res = self.forward(source, padding_mask, mask=mask, features_only=True)
        return res["x"], res["padding_mask"]

    def get_logits(self, net_output):
        logits = net_output["x"]
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    def get_targets(self, sample, net_output, expand_steps=True):
        x = net_output["x"]
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    def get_extra_losses(self, net_output):
        pen = []

        if "prob_perplexity" in net_output:
            pen.append(
                (net_output["num_vars"] - net_output["prob_perplexity"])
                / net_output["num_vars"]
            )

        if "features_pen" in net_output:
            pen.append(net_output["features_pen"])

        return pen

    def remove_pretraining_modules(self):
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None


@register_model("siamese_wav2vec2")
class SiameseWav2VecModel2(Wav2Vec2Model):
    def __init__(self, args):
        super().__init__(args)

    def forward(self, source, padding_mask=None, mask=True, features_only=False, target=None):
        assert target is not None

        source_features, source_features_pen = self.compute_features(source, self.feature_grad_mult, for_target=False)
        target_features, target_features_pen = self.compute_features(target, self.feature_grad_mult, for_target=True)
        features_pen = self.compute_penalty(source_features_pen, target_features_pen)

        if padding_mask is not None:
            extra = padding_mask.size(1) % source_features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), source_features.size(1), -1)
            padding_mask = padding_mask.all(-1)

        if self.post_extract_proj is not None:
            source_features = self.post_extract_proj(source_features)

        source_features = self.dropout_input(source_features)
        target_features = self.dropout_features(target_features)

        if mask:
            x, mask_indices = self.apply_mask(source_features, padding_mask)
            if mask_indices is not None:
                y = target_features[mask_indices].view(target_features.size(0), -1, target_features.size(-1))
            else:
                y = target_features
        else:
            x = source_features
            y = target_features
            mask_indices = None

        # x = self.encoder(x, padding_mask=padding_mask)
        x = self.apply_encoder(x, padding_mask)

        if features_only:
            return {"x": x, "padding_mask": padding_mask}

        y = self.apply_encoder(y, padding_mask)

        if self.quantizer:
            raise NotImplementedError
            # q = self.quantizer(y, produce_targets=False)
            # y = q["x"]
            # num_vars = q["num_vars"]
            # code_ppl = q["code_perplexity"]
            # prob_ppl = q["prob_perplexity"]
            # curr_temp = q["temp"]
            #
            # y = self.project_q(y)
            #
            # if self.negatives_from_everywhere:
            #     neg_cands, *_ = self.quantizer(unmasked_features, produce_targets=False)
            #     negs, _ = self.sample_negatives(neg_cands, y.size(1))
            #     negs = self.project_q(negs)
            #
            # else:
            #     negs, _ = self.sample_negatives(y, y.size(1))
            #
            # if self.codebook_negatives > 0:
            #     cb_negs = self.quantizer.sample_from_codebook(
            #         y.size(0) * y.size(1), self.codebook_negatives
            #     )
            #     cb_negs = cb_negs.view(
            #         self.codebook_negatives, y.size(0), y.size(1), -1
            #     )  # order doesnt matter
            #     cb_negs = self.project_q(cb_negs)
            #     negs = torch.cat([negs, cb_negs], dim=0)
        else:
            y = self.project_q(y)

            if self.negatives_from_everywhere:
                negs, _ = self.sample_negatives(y, y.size(1))
                negs = self.project_q(negs)
            else:
                negs, _ = self.sample_negatives(y, y.size(1))

        x = x[mask_indices].view(x.size(0), -1, x.size(-1))

        if self.target_glu:
            y = self.target_glu(y)
            negs = self.target_glu(negs)

        x = self.final_proj(x)
        x = self.compute_preds(x, y, negs)

        result = {"x": x, "padding_mask": padding_mask, "features_pen": features_pen}

        # if prob_ppl is not None:
        #     result["prob_perplexity"] = prob_ppl
        #     result["code_perplexity"] = code_ppl
        #     result["num_vars"] = num_vars
        #     result["temp"] = curr_temp

        return result


@register_model("wav2vec2_tracking")
class Wav2Vec2TrackingModel(Wav2Vec2Model):
    """
    Has a BYOL style tracking feature extractor on the target side
    """

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        Wav2Vec2Model.add_args(parser)

        parser.add_argument(
            "--tracking-tau", default=0.99, type=float
        )

    def __init__(self, args):
        super().__init__(args)

        self.tracking_feature_extractor = copy.deepcopy(self.feature_extractor)
        self.tracking_layer_norm = copy.deepcopy(self.layer_norm)
        self.tracking_tau = args.tracking_tau
        self.tracking_encoder = copy.deepcopy(self.encoder)

    def apply_encoder_to_target(self, padding_mask, y):
        with torch.no_grad():
            if self.args.mask_target:
                y = self.tracking_encoder(y, padding_mask=padding_mask)
            else:
                y = self.tracking_encoder(y)
        self.update_tracking(self.encoder, self.tracking_encoder)
        return y

    def compute_features(self, source, feature_grad_mult, for_target=False):
        if not for_target:
            return super(Wav2Vec2TrackingModel, self).compute_features(source, feature_grad_mult, for_target)
        else:
            with torch.no_grad():
                features = self.tracking_feature_extractor(source)
            features_pen = 0
            features = features.transpose(1, 2)
            features = self.tracking_layer_norm(features)
            self.update_tracking(self.feature_extractor, self.tracking_feature_extractor)
            self.update_tracking(self.layer_norm, self.tracking_layer_norm)
            return features, features_pen

    def update_tracking(self, module1: nn.Module, module2: nn.Module):
        with torch.no_grad():
            for name, param1 in module1.named_parameters():
                param2 = getattr(module2, name)
                param2 = param1 * self.tracking_tau + param2 * (1 - self.tracking_tau)
                setattr(module2, name, param2)

    def compute_penalty(self, pen1, pen2):
        return pen1


class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        in_d: int = 1,
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        if x.dim() == 2:
            # BxT -> BxCxT
            x = x.unsqueeze(1)
        else:
            # BxTxC -> BxCxT
            x = x.permute(0, 2, 1).contiguous()

        for conv in self.conv_layers:
            x = conv(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())

        self.conformer_list = None
        if getattr(args, "conformer_list", None):
            self.conformer_list = eval(getattr(args, "conformer_list", "None"))

        self.layers = nn.ModuleList(
            [
                self.create_transformer_layer(args, i)
                for i in range(args.encoder_layers)
            ]
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def create_transformer_layer(self, args, lyrnum=None):
        if self.conformer_list:
            trans_type = self.conformer_list[lyrnum]
        else:
            trans_type = args.transformer_type

        if trans_type == 'transformer':
            layer = TransformerSentenceEncoderLayer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=self.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                activation_fn=args.activation_fn,
                layer_norm_first=args.layer_norm_first,
            )
        elif trans_type == 'conformer':
            use_rel_posn_mha = args.use_rel_posn_mha
            if use_rel_posn_mha and args.rel_posn_mha_list is not None:
                rel_posn_mha_list = eval(args.rel_posn_mha_list)
                assert len(rel_posn_mha_list) == args.encoder_layers
                use_rel_posn_mha = rel_posn_mha_list[lyrnum]

            use_mha = True
            if args.conformer_mha_list is not None:
                conformer_mha_list = eval(args.conformer_mha_list)
                assert len(conformer_mha_list) == args.encoder_layers, f'{conformer_mha_list} {args.encoder_layers}'
                use_mha = conformer_mha_list[lyrnum]

            layer = ConformerEncoderLayer(
                embedding_dim=self.embedding_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_fn1=args.activation_fn1,
                activation_fn2=args.activation_fn2,
                kern_size=args.conformer_kernel_size,
                ffn_scale=args.ffn_scale,
                no_expand_ffn=args.no_expand_ffn,
                use_rel_posn_mha=use_rel_posn_mha,
                num_relpos_embeds=args.num_relpos_embeds,
                lin_dropout=args.lin_dropout,
                use_mha=use_mha,
            )
        else:
            raise Exception(f"Invalid transformer type: {trans_type}")

        return layer

    def forward(self, x, padding_mask=None):
        x = self.extract_features(x, padding_mask)

        if self.layer_norm_first:
            x = self.layer_norm(x)

        return x

    def extract_features(self, x, padding_mask=None):

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x += x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=False)
                layer_results.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn


class ConformerEncoderLayer(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 768,
            num_attention_heads: float = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_fn1: str = "swish",
            activation_fn2: str = "gelu",
            kern_size: int = 32,
            ffn_scale: float = 0.5,
            no_expand_ffn: bool =False,
            use_rel_posn_mha: bool =False,
            num_relpos_embeds: int =768,
            lin_dropout: float =0.1,
            use_mha: bool = True
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.kern_size = kern_size
        self.ffn_scale = ffn_scale
        self.use_rel_posn_mha = use_rel_posn_mha
        self.use_mha = use_mha

        # pad = (kern_size + 1) // 2
        pad = kern_size // 2
        self.activation_fn1 = utils.get_activation_fn(activation_fn1)
        self.activation_fn2 = utils.get_activation_fn(activation_fn2)
        embedding_dim_expand = embedding_dim if no_expand_ffn else embedding_dim * 4
        self.ff1 = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim_expand),
            Lambda(self.activation_fn1),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim_expand, embedding_dim),
            nn.Dropout(dropout),
        )
        if self.use_mha:
            self.self_attn_layer_norm = nn.LayerNorm(embedding_dim)
            if use_rel_posn_mha:
                self.self_attn = RelativePositionalMultiHeadAttention(
                    embedding_dim,
                    num_attention_heads,
                    num_relpos_embeds=num_relpos_embeds,
                    lin_dropout=lin_dropout,
                    att_dropout=attention_dropout,
                )
            else:
                self.self_attn = MultiheadAttention(
                    embedding_dim,
                    num_attention_heads,
                    dropout=attention_dropout,
                    self_attention=True,
                )
            self.self_attn_dropout = nn.Dropout(dropout)
        self.conv = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            Permute(1, 2, 0),  # T x B x C -> B x C x T
            nn.Conv1d(embedding_dim, embedding_dim * 2, kernel_size=1),
            Lambda(self.activation_fn2),
            nn.Conv1d(embedding_dim * 2, embedding_dim * 2, kern_size,
                      groups=embedding_dim, padding=pad),
            SamePad(kern_size),
            nn.BatchNorm1d(embedding_dim * 2),
            Lambda(self.activation_fn1),
            nn.Conv1d(embedding_dim * 2, embedding_dim, kernel_size=1),
            nn.Dropout(dropout),
            Permute(2, 0, 1),  # B x C x T -> T x B x C
        )
        self.ff2 = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim_expand),
            Lambda(self.activation_fn1),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim_expand, embedding_dim),
            nn.Dropout(dropout),
        )
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
            need_weights: bool = False,
            att_args=None,
    ):
        # T x B x C
        x = x + self.ffn_scale * self.ff1(x)
        residual = x
        # T x B x C
        if self.use_mha:
            x = self.self_attn_layer_norm(x)
            if self.use_rel_posn_mha:
                x, attn = self.self_attn(
                    x, self_attn_padding_mask=self_attn_padding_mask
                )
            else:
                x, attn = self.self_attn(
                    query=x,
                    key=x,
                    value=x,
                    key_padding_mask=self_attn_padding_mask,
                    need_weights=need_weights,
                )
            x = residual + self.self_attn_dropout(x)
        else:
            attn = None
        x = x + self.conv(x)
        x = x + self.ffn_scale * self.ff2(x)
        x = self.final_layer_norm(x)
        return x, attn


@register_model_architecture("wav2vec2", "wav2vec2")
def base_architecture(args):
    args.extractor_mode = getattr(args, "extractor_mode", "default")

    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)

    args.activation_fn = getattr(args, "activation_fn", "gelu")

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)

    args.final_dim = getattr(args, "final_dim", 0)

    args.layer_norm_first = getattr(args, "layer_norm_first", False)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)

    conv_feature_layers = "[(512, 10, 5)]"
    conv_feature_layers += " + [(512, 8, 4)]"
    conv_feature_layers += " + [(512, 4, 2)] * 3"
    conv_feature_layers += " + [(512, 1, 1)]"
    args.conv_feature_layers = getattr(args, "conv_feature_layers", conv_feature_layers)

    args.logit_temp = getattr(args, "logit_temp", 0.1)

    args.quantize_targets = getattr(args, "quantize_targets", False)
    args.quantize_input = getattr(args, "quantize_input", False)

    args.feature_grad_mult = getattr(args, "feature_grad_mult", 1.0)

    args.latent_vars = getattr(args, "latent_vars", 320)
    args.latent_groups = getattr(args, "latent_groups", 2)
    args.latent_dim = getattr(args, "latent_dim", 0)

    args.mask_length = getattr(args, "mask_length", 10)
    args.mask_prob = getattr(args, "mask_prob", 0.65)
    args.mask_selection = getattr(args, "mask_selection", "static")
    args.mask_other = getattr(args, "mask_other", 0)
    args.no_mask_overlap = getattr(args, "no_mask_overlap", False)
    args.mask_min_space = getattr(args, "mask_min_space", 1)

    args.mask_channel_length = getattr(args, "mask_channel_length", 10)
    args.mask_channel_prob = getattr(args, "mask_channel_prob", 0)
    args.mask_channel_selection = getattr(args, "mask_channel_selection", "static")
    args.mask_channel_other = getattr(args, "mask_channel_other", 0)
    args.no_mask_channel_overlap = getattr(args, "no_mask_channel_overlap", False)
    args.mask_channel_min_space = getattr(args, "mask_channel_min_space", 1)

    args.dropout_input = getattr(args, "dropout_input", 0)
    args.dropout_features = getattr(args, "dropout_features", 0)

    args.num_negatives = getattr(args, "num_negatives", 100)
    args.negatives_from_everywhere = getattr(args, "negatives_from_everywhere", False)
    args.cross_sample_negatives = getattr(args, "cross_sample_negatives", 0)
    args.codebook_negatives = getattr(args, "codebook_negatives", 0)

    args.conv_pos = getattr(args, "conv_pos", 128)
    args.conv_pos_groups = getattr(args, "conv_pos_groups", 16)

    args.latent_temp = getattr(args, "latent_temp", "(2,0.5,0.999995)")

    args.target_glu = getattr(args, "target_glu", False)

    args.conv_bias = getattr(args, "conv_bias", False)


@register_model_architecture("wav2vec2_tracking", "wav2vec2_tracking")
def base_architecture_tracking(args):
    base_architecture(args)
    args.tracking_tau = getattr(args, "tracking_tau", 0.99)
