from logging import getLogger
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modules.layer_norm import LayerNorm
#from torch.nn import LayerNorm
from ..modules.multihead_attention import MultiheadAttention
from ..modules.sinusoidal_positional_embedding import SinusoidalPositionalEmbedding
from ..sequence_generator import SequenceGenerator

from . import LatentState


logger = getLogger()


class TransformerEncoder(nn.Module):
    """Transformer encoder."""

    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.n_words = args.src_n_words
        embed_dim = args.encoder_embed_dim
        self.embeddings = Embedding(self.n_words, embed_dim, padding_idx=args.pad_index)
        self.padding_idx = args.pad_index
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = PositionalEmbedding(
            1024, embed_dim, self.padding_idx,
            left_pad=args.left_pad_source,
        )
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(args)
            for _ in range(args.encoder_layers)])

    def forward(self, src_tokens, src_lengths):
        # embed tokens and positions
        x = self.embed_scale * self.embeddings(src_tokens)
        x += self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # compute padding mask
        encoder_padding_mask = src_tokens.t().eq(self.padding_idx)

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)

        return LatentState(
            input_len=src_lengths,
            dec_input={
                'encoder_out': x,  # T x B x C
                'encoder_padding_mask': encoder_padding_mask,  # B x T
            }
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()

    @staticmethod
    def expand_encoder_out_(encoder_out, beam_size):
        T, B, C = encoder_out['encoder_out'].size()
        assert encoder_out['encoder_padding_mask'].size() == (B, T)
        encoder_out['encoder_out'] = encoder_out['encoder_out'].repeat(1, 1, beam_size).view(T, -1, C)
        encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].repeat(1, beam_size).view(-1, T)


class TransformerDecoder(nn.Module):
    """Transformer decoder."""
    def __init__(self, args, encoder):
        super().__init__()
        self.dropout = args.dropout
        self.n_words = args.tgt_n_words
        self.share_encdec_emb = args.share_encdec_emb
        self.share_decpro_emb = args.share_decpro_emb
        self.encoder_class = encoder.__class__
        self.beam_size = args.beam_size
        self.length_penalty = args.length_penalty

        # indexes
        self.eos_index = args.eos_index
        self.pad_index = args.pad_index
        self.bos_index = args.bos_index

        # embedding layers
        self.emb_dim = args.decoder_embed_dim
        if self.share_encdec_emb:
            logger.info("Sharing encoder and decoder input embeddings")
            embeddings = encoder.embeddings
        else:
            embeddings = Embedding(self.n_words, self.emb_dim, padding_idx=args.pad_index)
        self.embeddings = embeddings
        self.embed_scale = math.sqrt(self.emb_dim)
        self.embed_positions = PositionalEmbedding(
            1024, self.emb_dim, self.pad_index,
            left_pad=args.left_pad_target,
        )
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(args)
            for _ in range(args.decoder_layers)])

        # projection layers
        proj = Linear(self.emb_dim, self.n_words)
        if self.share_decpro_emb:
            logger.info("Sharing input embeddings and projection matrix in the decoder")
            proj.weight = self.embeddings.weight  # 这里通过赋值weight 实际上是将指针赋给了相应对象 二者共享变量 实际上是一个weight
        self.proj = proj

    def forward(self, encoded, y, one_hot=False, incremental_state=None):
        assert not one_hot, 'one_hot=True has not been implemented for transformer'

        prev_output_tokens = y  # T x B
        encoder_out = encoded.dec_input
        embed_tokens = self.embeddings
        proj_layer = self.proj

        # embed positions
        positions = self.embed_positions(prev_output_tokens, incremental_state)

        # embed tokens and positions
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[-1:, :]  # only keep last time step
        x = self.embed_scale * embed_tokens(prev_output_tokens)
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        # decoder layers
        for layer in self.layers:
            x, attn = layer(
                x,
                encoder_out['encoder_out'],
                encoder_out['encoder_padding_mask'],
                incremental_state=incremental_state,
            )

        # project back to size of vocabulary
        x = proj_layer(x)
        return x

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def reorder_incremental_state_(self, incremental_state, new_order):
        """Reorder incremental state.

        This should be called when the order of the input has changed from the
        previous time step. A typical use case is beam search, where the input
        order changes between time steps based on the selection of beams.
        """
        def apply_reorder_incremental_state(module):
            if module != self and hasattr(module, 'reorder_incremental_state'):
                module.reorder_incremental_state(
                    incremental_state,
                    new_order,
                )
        self.apply(apply_reorder_incremental_state)

    def reorder_encoder_out_(self, encoder_out_dict, new_order):
        if encoder_out_dict['encoder_padding_mask'] is not None:
            encoder_out_dict['encoder_padding_mask'] = \
                encoder_out_dict['encoder_padding_mask'].index_select(0, new_order)

    def generate(self, encoded, max_len=200, sample=False, temperature=None):
        """
        Generate a sentence from a given initial state.
        Input:
            - FloatTensor of size (batch_size, hidden_dim) representing
              sentences encoded in the latent space
        Output:
            - LongTensor of size (seq_len, batch_size), word indices
            - LongTensor of size (batch_size,), sentence x_len
        """
        if self.beam_size > 0:
            return self.generate_beam(encoded, self.beam_size, max_len, sample, temperature)
        else:
            logger.error('Not implemented! Beam size need a value > 0 !')
            exit(0)

    def generate_beam(self, encoded, beam_size=20, max_len=100, sample=False, temperature=None):
        """
        Generate a sentence from a given initial state.
        Input:
            - FloatTensor of size (batch_size, hidden_dim) representing
              sentences encoded in the latent space
        Output:
            - LongTensor of size (seq_len, batch_size), word indices
            - LongTensor of size (batch_size,), sentence x_len
        """
        self.encoder_class.expand_encoder_out_(encoded.dec_input, beam_size)

        x_len = encoded.input_len
        is_cuda = encoded.dec_input['encoder_out'].is_cuda
        one_hot = None

        # check inputs
        # assert latent.size() == (x_len.max(), x_len.size(0) * beam_size, self.emb_dim)
        assert (sample is True) ^ (temperature is None)
        assert temperature is None, 'not supported'

        generator = SequenceGenerator(
            self, self.bos_index, self.pad_index, self.eos_index,
            self.n_words, beam_size=beam_size, maxlen=max_len, sampling=sample,
            len_penalty=self.length_penalty,
        )
        if is_cuda:
            x_len = x_len.cuda()
        results = generator.generate(x_len, encoded)

        lengths = torch.LongTensor([sent[0]['tokens'].numel() for sent in results])
        lengths.add_(1)  # for BOS
        max_len = lengths.max()
        bsz = len(results)
        decoded = results[0][0]['tokens'].new(max_len, bsz).fill_(0)
        decoded[0, :] = self.bos_index
        for i, sent in enumerate(results):
            ntoks = sent[0]['tokens'].numel()  # pick the top beam result
            decoded[1:ntoks + 1, i] = sent[0]['tokens']

        return decoded, lengths, one_hot


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: dropout -> add residual -> layernorm.
    In the tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    dropout -> add residual.
    We default to the approach in the paper, but the tensor2tensor approach can
    be enabled by setting `normalize_before=True`.
    """
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

    def forward(self, x, encoder_padding_mask):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block."""
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.dropout = args.dropout
        self.relu_dropout = args.relu_dropout
        self.normalize_before = args.encoder_normalize_before
        self.encoder_attn = MultiheadAttention(
            self.embed_dim, args.decoder_attention_heads,
            dropout=args.attention_dropout,
        )
        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(3)])

    def forward(self, x, encoder_out, encoder_padding_mask, incremental_state=None):
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(
            query=x, key=x, value=x, mask_future_timesteps=True,
            incremental_state=incremental_state, need_weights=False,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x, attn = self.encoder_attn(
            query=x, key=encoder_out, value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            incremental_state=incremental_state, static_kv=True,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)

        residual = x
        x = self.maybe_layer_norm(2, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(2, x, after=True)
        return x, attn

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def PositionalEmbedding(num_embeddings, embedding_dim, padding_idx, left_pad):
    m = SinusoidalPositionalEmbedding(embedding_dim, padding_idx, left_pad, init_size=num_embeddings)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    nn.init.constant_(m.bias, 0.)
    return m
