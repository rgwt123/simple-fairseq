from logging import getLogger
from ..fairseq_utils import set_incremental_state, get_incremental_state
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..sequence_generator import SequenceGenerator
from . import LatentState


logger = getLogger()


class LSTMEncoder(nn.Module):
    """Transformer encoder."""

    def __init__(self, args):
        super().__init__()
        self.dropout_in = args.dropout
        self.dropout_out = args.dropout
        self.n_words = args.src_n_words
        self.embed_dim = args.encoder_embed_dim
        self.hidden_size = args.hidden_dim
        self.encoder_layers = args.encoder_layers
        self.embeddings = Embedding(self.n_words, self.embed_dim, padding_idx=args.pad_index)
        self.padding_idx = args.pad_index
        self.lstm = LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_size,
            num_layers=args.encoder_layers,
            dropout=args.dropout if args.encoder_layers > 1 else 0.,
            bidirectional=True,
        )
        self.output_units = self.hidden_size*2

    def forward(self, src_tokens, src_lengths):
        # embed tokens and positions
        seq_len, bsz = src_tokens.size()
        x = self.embeddings(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths)
        # compute padding mask
        encoder_padding_mask = src_tokens.t().eq(self.padding_idx)
        state_size = self.encoder_layers*2, bsz, self.hidden_size
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0,c0))
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_idx)
        x = F.dropout(x, p=self.dropout_out, training=self.training)

        assert list(x.size()) == [seq_len, bsz, self.output_units]

        def combine_bidir(outs):
            return outs.view(self.encoder_layers, 2, bsz, -1).transpose(1,2).contiguous().view(self.encoder_layers, bsz, -1)

        final_hiddens = combine_bidir(final_hiddens)
        final_cells = combine_bidir(final_cells)


        return LatentState(
            input_len=src_lengths,
            dec_input={
                'encoder_out': (x, final_hiddens, final_cells),  # T x B x C
                'encoder_padding_mask': encoder_padding_mask,  # B x T
            }
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()

    @staticmethod
    def expand_encoder_out_(encoder_out, beam_size):
        T, B, C = encoder_out['encoder_out'][0].size()
        assert encoder_out['encoder_padding_mask'].size() == (B, T)
        encoder_out['encoder_out'] = tuple(
            eo.repeat(1, 1, beam_size).view(eo.size(0), -1, eo.size(2))
            for eo in encoder_out['encoder_out']
        )
        encoder_out['encoder_padding_mask'] = encoder_out['encoder_padding_mask'].repeat(1, beam_size).view(-1, T)

class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, output_embed_dim):
        super().__init__()
        self.input_proj = Linear(input_embed_dim, output_embed_dim, bias=False)
        self.output_proj = Linear(input_embed_dim+output_embed_dim, output_embed_dim, bias=False)

    def forward(self, input, source_hids, encoder_padding_mask):
        x = self.input_proj(input)
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)
        encoder_padding_mask = encoder_padding_mask.transpose(0,1)

        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')).type_as(attn_scores)

        attn_scores = F.softmax(attn_scores, dim=0)

        x = (attn_scores.unsqueeze(2)*source_hids).sum(dim=0)
        # source_hids [srclen, bsz, 1024] attn_scores [srclen, bsz]
        # x [bsz, 1024]

        x = F.tanh(self.output_proj(torch.cat((x, input),dim=1)))

        return x, attn_scores

class LSTMDecoder(nn.Module):
    """LSTM decoder."""
    def __init__(self, args, encoder):
        super().__init__()
        self.beam_size = args.beam_size
        self.encoder_class = encoder.__class__
        self.length_penalty = args.length_penalty
        self.dropout_in = args.dropout
        self.dropout_out = args.dropout
        self.n_words = args.tgt_n_words
        self.embed_dim = args.decoder_embed_dim
        self.hidden_size = args.hidden_dim*2
        self.decoder_layers = args.decoder_layers
        self.embeddings = Embedding(self.n_words, self.embed_dim, padding_idx=args.pad_index)

        # indexes
        self.eos_index = args.eos_index
        self.pad_index = args.pad_index
        self.bos_index = args.bos_index

        # model
        self.encoder_output_units = self.hidden_size
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=self.encoder_output_units + self.embed_dim if layer==0 else self.hidden_size,
                hidden_size=self.hidden_size
            )
            for layer in range(self.decoder_layers)
        ])
        self.attention = AttentionLayer(self.encoder_output_units, self.hidden_size)

        if self.hidden_size != self.embed_dim:
            self.additional_fc = Linear(self.hidden_size, self.embed_dim)


    def forward(self, encoded, y, incremental_state=None):
        prev_output_tokens = y  # T x B
        encoder_out = encoded.dec_input['encoder_out']
        encoder_padding_mask = encoded.dec_input['encoder_padding_mask']
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[-1:,:]
        seq_len, bsz = prev_output_tokens.size()

        encoder_outs, _, _ = encoder_out[:3]
        srclen = encoder_outs.size(0)

        x = self.embeddings(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        cache_state = get_incremental_state(self, incremental_state, 'cached_state')
        if cache_state is not None:
            prev_hiddens, prev_cells, input_feed = cache_state
        else:
            _, encoder_hiddens, encoder_cells = encoder_out[:3]
            # [srclen, bsz, 1024] [encoder_layer, bsz, 1024] [encoder_layer, bsz, 1024]
            prev_hiddens = [encoder_hiddens[i] for i in range(self.decoder_layers)]
            prev_cells = [encoder_cells[i] for i in range(self.decoder_layers)]
            input_feed = x.data.new(bsz, self.encoder_output_units).zero_()


        attn_scores = x.data.new(srclen, seq_len, bsz).zero_()
        outs = []
        for j in range(seq_len):
            input = torch.cat((x[j,:,:],input_feed), dim=1)

            for i,rnn in enumerate(self.layers):
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)
                # input [bsz, 1024]
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention
            # hidden [bsz, 1024] encoder_outs [srclen, bsz, 1024] padding [bsz, seqlen]
            out, attn_scores[:,j,:] = self.attention(hidden, encoder_outs, encoder_padding_mask)
            out = F.dropout(out, p =self.dropout_out, training=self.training)
            input_feed = out
            outs.append(out)

        set_incremental_state(
            self, incremental_state, 'cached_state', (prev_hiddens, prev_cells, input_feed)
        )

        x = torch.cat(outs, dim=0).view(seq_len, bsz, self.hidden_size)
        x = self.additional_fc(x)
        x = F.linear(x, self.embeddings.weight)

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
        cached_state = get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        set_incremental_state(self, incremental_state, 'cached_state', new_state)

    def reorder_encoder_out_(self, encoder_out_dict, new_order):
        encoder_out_dict['encoder_out'] = tuple(
            eo.index_select(1, new_order)
            for eo in encoder_out_dict['encoder_out']
        )
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
        is_cuda = encoded.dec_input['encoder_out'][0].is_cuda
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


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1,0.1)
    return m

def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1,0.1)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    m.weight.data.uniform_(-0.1,0.1)
    if bias:
        m.bias.data.uniform_(-0.1,0.1)
    return m
