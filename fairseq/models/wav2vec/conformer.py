from typing import Dict, Optional

import torch

from fairseq.modules import LayerNorm, FairseqDropout
from torch import nn, Tensor

from fairseq import utils, tasks
from fairseq.models import register_model, FairseqEncoderDecoderModel, FairseqEncoder, register_model_architecture, \
    FairseqIncrementalDecoder, BaseFairseqModel
from fairseq.models.wav2vec import ConvFeatureExtractionModel, TransformerEncoder
import torch.nn.functional as F


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    # nn.init.xavier_uniform_(m.weight)
    # if bias:
    #     nn.init.constant_(m.bias, 0.0)
    return m


def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    # for name, param in m.named_parameters():
    #     if 'weight' in name or 'bias' in name:
    #         param.data.uniform_(-0.1, 0.1)
    return m


@register_model("conformer_ctc")
class ConformerCtcModel(BaseFairseqModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--extractor-mode",
            choices=["default", "layer_norm"],
            help="mode for feature extractor. default has a single group norm with d groups in the first conv block, whereas layer_norm has layer norms in every block (meant to use with --normalize)",
        )

        # Encoder Args
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
            "--in-d", type=int, default=1, help="number of input channels"
        )

        # Conformer Arguments
        parser.add_argument(
            "--transformer-type",
            choices=('conformer',),
            help="whether to use conformer or transformer layers",
            default="conformer"
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
            '--share-decoder-input-output-embed',
            default=True,
            action='store_true',
        )


    def __init__(self, args, encoder):
        super().__init__()
        self.encoder = encoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_ctc_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        encoder = cls.build_encoder(args)
        return cls(args, encoder)

    @classmethod
    def build_encoder(cls, args):
        return ConformerEncoder(args)

    def forward(self, **kwargs):
        encoder_out, padding_mask = self.encoder(tbc=False, **kwargs)
        return {
            'encoder_out': encoder_out.transpose(0, 1),
            'encoder_padding_mask': padding_mask,
            'padding_mask': padding_mask
        }

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict


@register_model("conformer_seq2seq")
class ConformerSeq2SeqModel(FairseqEncoderDecoderModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--extractor-mode",
            choices=["default", "layer_norm"],
            help="mode for feature extractor. default has a single group norm with d groups in the first conv block, whereas layer_norm has layer norms in every block (meant to use with --normalize)",
        )

        # Decoder Args
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the decoder",
        )
        parser.add_argument(
            "--decoder-attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights inside the decoder",
        )
        parser.add_argument(
            "--decoder-activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN inside the decoder",
        )
        parser.add_argument(
            "--lstm-hidden-size",
            default=1024,
            type=int,
        )
        parser.add_argument(
            "--num-lstm-layers",
            default=1,
            type=int,
        )
        # Encoder Args
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
            "--in-d", type=int, default=1, help="number of input channels"
        )

        # Conformer Arguments
        parser.add_argument(
            "--transformer-type",
            choices=('conformer',),
            help="whether to use conformer or transformer layers",
            default="conformer"
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
            '--share-decoder-input-output-embed',
            default=False,
            action='store_true',
        )


    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if not hasattr(args, "max_source_positions"):
            args.max_source_positions = 2048
        if not hasattr(args, "max_target_positions"):
            args.max_target_positions = 2048

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            return emb

        decoder_embed_tokens = build_embedding(tgt_dict, args.decoder_embed_dim)

        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args):
        return ConformerEncoder(args)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return LSTMDecoder(
            tgt_dict,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.lstm_hidden_size,
            out_embed_dim=embed_tokens.embedding_dim,
            num_layers=args.num_lstm_layers,
            dropout_in=args.decoder_dropout,
            dropout_out=args.decoder_dropout,
            encoder_output_units=args.encoder_embed_dim,
            share_input_output_embed=args.share_decoder_input_output_embed,
            max_target_positions=args.max_target_positions,
        )

    def forward(self, **kwargs):
        encoder_out, padding_mask = self.encoder(tbc=False, **kwargs)
        is_shape = None
        if 'incremental_state' in kwargs and kwargs['incremental_state']:
            is_shape = kwargs['incremental_state'].shape
        decoder_out = self.decoder(
            encoder_out=encoder_out,
            prev_output_tokens=kwargs['prev_output_tokens'].long(),
            incremental_state=kwargs.get('incremental_state', None),
            padding_mask=padding_mask,
        )
        return decoder_out

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict


class ConformerEncoder(FairseqEncoder):
    def __init__(self, args):
        self.args = args
        task = tasks.setup_task(args)
        super().__init__(task.source_dictionary)

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
            if self.embed != args.encoder_embed_dim
            else None
        )
        self.dropout_input = nn.Dropout(args.dropout_input)
        self.dropout_features = nn.Dropout(args.dropout_features)
        # final_dim = args.final_dim if args.final_dim > 0 else args.encoder_embed_dim
        self.encoder = TransformerEncoder(args)
        self.layer_norm = LayerNorm(self.embed)
        # d = w2v_args.encoder_embed_dim
        # self.proj = Linear(args.encoder_embed_dim, args.lstm_hidden_size)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):
        features = self.feature_extractor(source)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask.all(-1)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        features = self.dropout_input(features)
        features = self.encoder(features, padding_mask=padding_mask)
        # features = self.proj(features)
        return features, padding_mask

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


class LSTMDecoder(FairseqIncrementalDecoder):
    def __init__(
            self, dictionary, embed_dim=512, hidden_size=512,
            out_embed_dim=512, num_layers=1, dropout_in=0.1, dropout_out=0.1,
            encoder_output_units=512, share_input_output_embed=False,
            max_target_positions=1e5, residuals=False):
        super().__init__(dictionary)
        self.dropout_in_module = FairseqDropout(dropout_in, module_name=self.__class__.__name__)
        self.dropout_out_module = FairseqDropout(dropout_out, module_name=self.__class__.__name__)
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.max_target_positions = max_target_positions
        self.residuals = residuals
        self.num_layers = num_layers
        self.embed_dim = embed_dim

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        self.encoder_output_units = encoder_output_units

        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=hidden_size + embed_dim if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])
        self.attention = AttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=False)
        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)
        if not share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings)

    def forward(self, prev_output_tokens, encoder_out, padding_mask,
                incremental_state=None, **kwargs):
        x, attn_scores = self.extract_features(
            prev_output_tokens, encoder_out, padding_mask, incremental_state
        )
        return self.output_layer(x), attn_scores

    def output_layer(self, x):
        """Project features to the vocabulary size."""
        if self.share_input_output_embed:
            x = F.linear(x, self.embed_tokens.weight)
        else:
            x = self.fc_out(x)
        return x


    def extract_features(self, prev_output_tokens, encoder_outs, padding_mask,
                         incremental_state=None):
        encoder_outs = encoder_outs.transpose(0, 1)  # BxTxC --> TxBxC
        srclen = encoder_outs.size(0)
        if incremental_state is not None and len(incremental_state) > 0:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout_in_module(x)
        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        if incremental_state is not None and len(incremental_state) > 0:
            prev_hiddens, prev_cells, input_feed = self.get_cached_state(incremental_state)
        else:
            prev_hiddens, prev_cells = [], []
            for i in range(self.num_layers):
                prev_hiddens.append(x.new_zeros(bsz, self.hidden_size))
                prev_cells.append(x.new_zeros(bsz, self.hidden_size))
            input_feed = x.new_zeros(bsz, self.hidden_size)
        attn_scores = x.new_zeros(srclen, seqlen, bsz)
        outs = []
        for j in range(seqlen):
            input = torch.cat((x[j, :, :], input_feed), dim=1)
            for i, rnn in enumerate(self.layers):
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))
                # hidden state becomes the input to the next layer
                input = self.dropout_out_module(hidden)
                if self.residuals:
                    input = input + prev_hiddens[i]

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell
            # apply attention using the last layer's hidden state
            out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs, padding_mask)
            out = self.dropout_out_module(out)
            input_feed = out
            outs.append(out)

        # Stack all the necessary tensors together and store
        prev_hiddens_tensor = torch.stack(prev_hiddens)
        prev_cells_tensor = torch.stack(prev_cells)
        cache_state = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": prev_hiddens_tensor,
                "prev_cells": prev_cells_tensor,
                "input_feed": input_feed,
            }
        )
        self.set_incremental_state(incremental_state, 'cached_state', cache_state)

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if hasattr(self, 'additional_fc'):
            x = self.additional_fc(x)
            x = self.dropout_out_module(x)
        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training:
            assert attn_scores is not None
            attn_scores = attn_scores.transpose(0, 2)
        return x, attn_scores

    def get_cached_state(self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]]):
        cached_state = self.get_incremental_state(incremental_state, 'cached_state')
        assert cached_state is not None
        prev_hiddens_ = cached_state["prev_hiddens"]
        assert prev_hiddens_ is not None
        prev_cells_ = cached_state["prev_cells"]
        assert prev_cells_ is not None
        prev_hiddens = [prev_hiddens_[i] for i in range(self.num_layers)]
        prev_cells = [prev_cells_[j] for j in range(self.num_layers)]
        input_feed = cached_state["input_feed"]  # can be None for decoder-only language models
        return prev_hiddens, prev_cells, input_feed

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        if incremental_state is None or len(incremental_state) == 0:
            return
        prev_hiddens, prev_cells, input_feed = self.get_cached_state(incremental_state)
        prev_hiddens = [p.index_select(0, new_order) for p in prev_hiddens]
        prev_cells = [p.index_select(0, new_order) for p in prev_cells]
        if input_feed is not None:
            input_feed = input_feed.index_select(0, new_order)
        cached_state_new = torch.jit.annotate(
            Dict[str, Optional[Tensor]],
            {
                "prev_hiddens": torch.stack(prev_hiddens),
                "prev_cells": torch.stack(prev_cells),
                "input_feed": input_feed,
            }
        )
        self.set_incremental_state(incremental_state, 'cached_state', cached_state_new),
        return

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.max_target_positions


class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x source_embed_dim

        # x: bsz x source_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

        # don't attend over padding
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.transpose(0, 1)
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


@register_model_architecture("conformer_seq2seq", "conformer_seq2seq")
def base_architecture(args):
    args.extractor_mode = getattr(args, "extractor_mode", "default")

    args.encoder_layers = getattr(args, "encoder_layers", 16)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)

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

    args.dropout_input = getattr(args, "dropout_input", 0)
    args.dropout_features = getattr(args, "dropout_features", 0)

    args.conv_pos = getattr(args, "conv_pos", 128)
    args.conv_pos_groups = getattr(args, "conv_pos_groups", 16)
    args.conv_bias = getattr(args, "conv_bias", False)

    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
    args.decoder_dropout = getattr(args, "decoder_dropout", 0)
    args.lstm_hidden_size = getattr(args, "lstm_hidden_size", 320)
    args.num_lstm_layers = getattr(args, "num_lstm_layers", 1)



@register_model_architecture("conformer_ctc", "conformer_ctc")
def base_ctc_architecture(args):
    args.extractor_mode = getattr(args, "extractor_mode", "default")

    args.encoder_layers = getattr(args, "encoder_layers", 16)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)

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

    args.dropout_input = getattr(args, "dropout_input", 0)
    args.dropout_features = getattr(args, "dropout_features", 0)

    args.conv_pos = getattr(args, "conv_pos", 128)
    args.conv_pos_groups = getattr(args, "conv_pos_groups", 16)
    args.conv_bias = getattr(args, "conv_bias", False)
