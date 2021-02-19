# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.models.wav2vec.wav2vec2 import ConvFeatureExtractionModel, TransformerEncoder
from fairseq.models.wav2vec.wav2vec2 import EXTRACTOR_MODE_CHOICES, MASKING_DISTRIBUTION_CHOICES
import math
from dataclasses import dataclass, field
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.modules import (
    GradMultiply,
    GumbelVectorQuantizer,
    LayerNorm,
)


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
                nn.BatchNorm1d(num_channels),
                Permute(0, 2, 1),  # (B, D, T) --> (B, T, D)
            )

    def forward(self, x):
        if self.bn is not None:
            return self.bn(x)
        return x


@dataclass
class SiameseWav2Vec2Config(FairseqDataclass):
    extractor_mode: EXTRACTOR_MODE_CHOICES = field(
        default="default",
        metadata={
            "help": "mode for feature extractor. default has a single group norm with d "
            "groups in the first conv block, whereas layer_norm has layer norms in "
            "every block (meant to use with normalize=True)"
        },
    )
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )

    # dropouts
    dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for the transformer"}
    )
    attention_dropout: float = field(
        default=0.1, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN"}
    )
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a tarnsformer layer"}
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )

    layer_norm_first: bool = field(
        default=False, metadata={"help": "apply layernorm first in the transformer"}
    )
    conv_feature_layers: str = field(
        default="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
        metadata={
            "help": "string describing convolutional feature extraction layers in form of a python list that contains "
            "[(dim, kernel_size, stride), ...]"
        },
    )
    conv_bias: bool = field(
        default=False, metadata={"help": "include bias in conv encoder"}
    )
    quantize: bool = field(
        default=False, metadata={"help": "use quantized representations"}
    )
    feature_grad_mult: float = field(
        default=1.0, metadata={"help": "multiply feature extractor var grads by this"}
    )
    latent_vars: int = field(
        default=320,
        metadata={"help": "number of latent variables V in each group of the codebook"},
    )
    latent_groups: int = field(
        default=2,
        metadata={"help": "number of groups G of latent variables in the codebook"},
    )
    latent_dim: int = field(
        default=0,
        metadata={
            "help": "if > 0, uses this dimensionality for latent variables. "
            "otherwise uses final_dim / latent_groups"
        },
    )

    # masking
    mask_length: int = field(default=10, metadata={"help": "mask length"})
    mask_prob: float = field(
        default=0.65, metadata={"help": "probability of replacing a token with mask"}
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    mask_channel_min_space: int = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # positional embeddings
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )

    latent_temp: Tuple[float, float, float] = field(
        default=(2, 0.5, 0.999995),
        metadata={
            "help": "temperature for latent variable sampling. "
            "can be tuple of 3 values (start, end, decay)"
        },
    )

    # projection MLP
    projection_mlp_dim: int = field(
        default=768,
        metadata={"help": "Hidden dimension of projection mlp"}
    )
    projection_mlp_layers: int = field(
        default=3,
        metadata={"help": "Number of layers of projection mlp"}
    )
    final_dim: int = field(
        default=0,
        metadata={"help": "Output dimension. Set to encoder_embed_dim if it is <= 0"}
    )

    # prediction MLP
    prediction_mlp_dim: int = field(
        default=384,
        metadata={"help": "Hidden dimension of prediction mlp"}
    )
    prediction_mlp_layers: int = field(
        default=2,
        metadata={"help": "Number of layers of projection mlp"}
    )
    prediction_dim: int = field(
        default=0,
        metadata={
            "help": "Output dimension of prediction mlp."
            "Set to final_dim / encoder_embed_dim if it is <= 0"
        }
    )

    # Loss
    stop_gradient: bool = field(
        default=True,
        metadata={"help": "Apply stop gradient on the target side"}
    )

@register_model("siamese_wav2vec2", dataclass=SiameseWav2Vec2Config)
class SiameseWav2Vec2Model(BaseFairseqModel):
    def __init__(self, cfg: SiameseWav2Vec2Config):
        super().__init__()
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim and not cfg.quantize
            else None
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult
        self.stop_gradient = cfg.stop_gradient

        self.quantizer = None
        if cfg.quantize:
            vq_dim = cfg.latent_dim if cfg.latent_dim > 0 else cfg.encoder_embed_dim
            self.quantizer = GumbelVectorQuantizer(
                    dim=self.embed,
                    num_vars=cfg.latent_vars,
                    temp=cfg.latent_temp,
                    groups=cfg.latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                )
            self.project = nn.Linear(vq_dim, cfg.encoder_embed_dim)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        )

        self.encoder = TransformerEncoder(cfg)
        self.layer_norm = LayerNorm(self.embed)
        
        final_dim = cfg.final_dim if cfg.final_dim > 0 else cfg.encoder_embed_dim
        self.projection_mlp = self.build_mlp(
            cfg.encoder_embed_dim, final_dim, cfg.projection_mlp_dim,
            cfg.projection_mlp_layers
        )
        prediction_dim = cfg.prediction_dim if cfg.prediction_dim > 0 else final_dim
        self.prediction_mlp = self.build_mlp(
            final_dim, prediction_dim, cfg.prediction_mlp_dim,
            cfg.prediction_mlp_layers, output_act=False
        )

    def build_mlp(self, in_dim, out_dim, hidden_dim, num_layers, output_act=True):
        layers = [nn.Linear(in_dim, hidden_dim)]
        for _ in range(num_layers - 1):
            layers.extend([
                # nn.LayerNorm(hidden_dim),
                MaybeBatchNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ])
        layers.append(nn.Linear(hidden_dim, out_dim))
        if output_act:
            layers.extend([
                # nn.LayerNorm(out_dim),
                MaybeBatchNorm(out_dim),
                nn.SiLU(),
            ])
        return nn.Sequential(*layers)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

    @classmethod
    def build_model(cls, cfg: SiameseWav2Vec2Config, task=None):
        """Build a new model instance."""

        return cls(cfg)

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

    def compute_features1(self, source):
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        features_pen = features.float().pow(2).mean()
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        return features, features_pen

    def compute_features2(self, source, padding_mask, mask):
        features, features_pen = self.compute_features1(source)
        if padding_mask is not None:
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask.all(-1)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        features = self.dropout_features(features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None

        if self.quantizer:
            q = self.quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project(features)

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask)
        else:
            x = features
            mask_indices = None

        x = self.encoder(x, padding_mask=padding_mask)
        x = self.projection_mlp(x)
        result = {
            "features": x,
            "features_pen": features_pen,
            "mask_indices": mask_indices,
            "padding_mask": padding_mask,
            "prob_perplexity": prob_ppl,
            "code_perplexity": code_ppl,
            "num_vars": num_vars,
            "temp": curr_temp,
        }
        return result

    def forward(self, source, target=None, padding_mask=None, mask=True, features_only=False):
        assert features_only or target is not None, "target is required while pre-training"

        result1 = self.compute_features2(source, padding_mask, mask)

        if features_only:
            return {"x": result1["features"], "padding_mask": result1["padding_mask"]}

        # result2 = self.compute_features2(target, padding_mask, mask=False)  # TODO: Using same mask for source and target
        result2 = self.compute_features2(target, padding_mask, mask)

        x1 = result1["features"]
        x2 = result2["features"]
        mask_indices = result1["mask_indices"]
        x1 = x1[mask_indices].view(x1.size(0), -1, x1.size(-1))
        x2 = x2[mask_indices].view(x2.size(0), -1, x2.size(-1))
        y1 = self.prediction_mlp(x1)
        y2 = self.prediction_mlp(x2)

        result = {
            "x1": x1, "x2": x2, "y1": y1, "y2": y2,
            "features_pen": result1["features_pen"],
            "padding_mask": result1["padding_mask"],
        }
        if result1["prob_perplexity"] is not None:
            result.update({
                "prob_perplexity": result1["prob_perplexity"],
                "code_perplexity": result1["code_perplexity"],
                "num_vars": result1["num_vars"],
                "temp": result1["temp"],
            })
        return result

    def extract_features(self, source, padding_mask, mask=False):
        res = self.forward(source, padding_mask=padding_mask, mask=mask, features_only=True)
        return res["x"], res["padding_mask"]

    def get_features(self, net_output):
        x1, x2 = net_output["x1"], net_output["x2"]
        return x1.reshape(-1, x1.size(-1)), x2.reshape(-1, x2.size(-1))
    
    def get_predictions(self, net_output):
        y1, y2 = net_output["y1"], net_output["y2"]
        if self.stop_gradient:
            y1 = y1.detach()
            y2 = y2.detach()
        return y1.reshape(-1, y1.size(-1)), y2.reshape(-1, y2.size(-1))

    def get_extra_losses(self, net_output):
        pen = []

        # TODO: Loss on the other side???
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
        self.prediction_mlp = None


# @register_model("nce_siamese_wav2vec2", dataclass=SiameseWav2Vec2Config)
# class NceSiameseWav2Vec2Model(SiameseWav2Vec2Model):


@dataclass
class MomentumSiameseWav2Vec2Config(SiameseWav2Vec2Config):
    momentum: float = field(
        default=0.99, metadata={"help": "momentum param for the momentum encoder"}
    )


@register_model("momentum_siamese_wav2vec2", dataclass=MomentumSiameseWav2Vec2Config)
class SiameseWav2Vec2Model(SiameseWav2Vec2Model):
    def __init__(self, cfg: SiameseWav2Vec2Config):
        super().__init__(cfg)
        self.momentum = cfg.momentum
    


