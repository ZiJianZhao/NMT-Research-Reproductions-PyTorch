# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable

from xnmt.modules.embedding import Embedding
from xnmt.modules.attention import Attention
from xnmt.modules.stacked_rnn import StackedGRU, StackedLSTM
from xnmt.utils import aeq

class RNNDecoderBase(nn.Module):
    """
    RNN decoder base class
    Follow the implementation of 'RNNDecoderBase' in OpenNMT-py 
    Reference: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Models.py
    Model overview: First RNN, then Attention
    Reference paper:
        Effective Approaches to Attention-based Neural Machine Translation

    # ----------------------------------------------------------
    Note: the layers of decoder should be same as that of encoder.
    # ----------------------------------------------------------

    Args:
        rnn_type (str): type of RNN cell, one of [GRU, LSTM]
        num_layers (int): number of recurrent layers
        hidden_dim (int): the dimensionality of the hidden state `h`
        attn_type (str): type of attention, one of [mlp, dot, general]
        ctx_dim (int): the dimensionality of context vector
        embedding (nn.Module): predefined embedding module
        bidirectional_encoder (bool): whether the encoder is bidirectional
        dropout (float, optional): dropout
    """

    def __init__(self, rnn_type, num_layers, hidden_dim, attn_type, ctx_dim, 
            embedding, bidirectional_encoder=False, dropout=0.):

        super(RNNDecoderBase, self).__init__()
        
        self.rnn_type = rnn_type
        self.attn_type = attn_type
        self.bidirectional_encoder = bidirectional_encoder
        if bidirectional_encoder:
            self.encoder_directions = 2
        else:
            self.encoder_directions = 1
        if self.rnn_type not in ["GRU", "LSTM"]:
            raise Exception("Unsupported RNN Type in decoder")
        self.hidden_dim = hidden_dim

        self.embedding = embedding
        
        self.rnn = self._build_rnn(rnn_type, self._input_dim, hidden_dim, num_layers, dropout)
        
        self.attn_func = Attention(attn_type, ctx_dim, hidden_dim)
        
        self.linear_out = nn.Linear(ctx_dim+hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        self.init_params()

    def forward(self, tgt, ctx, state, ctx_lengths=None):
        """
        Applies a multi-layer RNN with following attention to an target sequence.

        Args:
            tgt (batch_size, seq_len): tensor containing the features of the target sequence.
            ctx (batch_size, ctx_len, ctx_dim): tensor containing context.
            state (RNNDecoderState): decoder initial state.
            ctx_lengths (LongTensor): [batch_size] containing context lengths.

        Returns: output, hidden
            outputs (batch_size * seq_len, hidden_dim): outputs of the decoder. 
            state  (RNNDecoderState): final hidden states.
        """
        
        assert isinstance(state, RNNDecoderState)

        batch_size, seq_len = tgt.size() 

        # outputs: (batch_size, seq_len, hidden_dim)
        outputs, hidden = self._run_forward_pass(tgt, ctx, state, ctx_lengths)

        # update state
        final_output = outputs[:, -1, :]
        state.update_state(hidden, final_output.unsqueeze(0))

        # return outputs
        outputs = outputs.contiguous().view(batch_size * seq_len, -1)

        return outputs, state

    def init_states(self, context, enc_states):
        """
        Initialize the hidden states decoder. 
        Following the implementation choice in OpenNMT-py:
            - decoder_hidden_dim = encoder_num_directions * encoder_hidden_dim

        Args:
            context (Variable): batch_size * enc_len * (enc_directions * enc_size)
            enc_states (tensor or tuple): (layers* directions, batch_size, enc_hidden_dim),

        Returns:
            dec_states (RNNDecoderState): initial decoder state
        """

        def fix_state(h):
            if self.bidirectional_encoder:
                h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            return h
        
        if isinstance(enc_states, tuple):
            aeq(enc_states[0].size(2) * self.encoder_directions, self.hidden_dim)
        else:
            aeq(enc_states.size(2) * self.encoder_directions, self.hidden_dim)

        if isinstance(enc_states, tuple):  # the encoder is a LSTM
            if self.rnn_type == 'GRU':
                hidden = fix_state(enc_states[0])
                return RNNDecoderState(context, self.hidden_dim, hidden)
            elif self.rnn_type == 'LSTM':
                hidden = tuple([fix_state(h) for h in enc_states])
                return RNNDecoderState(context, self.hidden_dim, hidden)
        else:
            if self.rnn_type == 'GRU':
                hidden = fix_state(enc_states)
                return RNNDecoderState(context, self.hidden_dim, hidden)
            else:
                raise Exception("Unsupported structure: encoder-gru --> decoder-lstm")

    def init_params(self):
        """
        This initializaiton is especially for gru according to the paper:
            Neural Machine Translation by Jointly Learning to Align and Translate

        Note: We just keep this for possible use. But we will initial all parameters in the outer loop.
        """
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.normal(param, 0, 0.01)
            elif 'weight_hh' in name:
                for i in range(0, param.data.size(0), self.hidden_dim):
                    nn.init.orthogonal(param.data[i:i+self.hidden_dim])
            elif 'bias' in name:
                nn.init.constant(param, 0)


class StdRNNDecoder(RNNDecoderBase):
    """
    Standard RNN decoder with attention.
    """

    def _run_forward_pass(self, tgt, ctx, state, ctx_lengths):
        """
        Applies a multi-layer RNN with following attention to an target sequence.

        Args:
            tgt (batch_size, seq_len): tensor containing the features of the target sequence.
            ctx (batch_size, ctx_len, ctx_dim): tensor containing context.
            state (RNNDecoderState): decoder initial state.
            ctx_lengths (LongTensor): [batch_size] containing context lengths.

        Returns: 
            outputs (batch_size, seq_len, hidden_dim): outputs of the decoder
            hidden (FloatTensor): tensor or tuple containing last hidden states.
        """
        if self.rnn_type.lower() == 'gru':
            hidden = state.hidden[0]
        elif self.rnn_type.lower() == 'lstm':
            hidden = state.hidden

        batch_size, seq_len = tgt.size()
        emb = self.embedding(tgt) # batch_size * seq_len * emb_dim

        outputs, hidden = self.rnn(emb, hidden)

        attns, ctxs = self.attn_func(outputs, ctx, ctx_lengths)

        outputs = torch.cat([ctxs, outputs], 2)
        outputs = self.linear_out(outputs)
        if self.attn_type.lower() in ['general', 'dot']:
            outputs = self.tanh(outputs)
        outputs = self.dropout(outputs)

        return outputs, hidden

    def _build_rnn(self, rnn_type, input_dim, hidden_dim, num_layers, dropout):
        return getattr(nn, rnn_type)(input_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout)
    
    @property
    def _input_dim(self):
        return self.embedding.emb_dim


class InputFeedRNNDecoder(RNNDecoderBase):
    """
    Standard RNN decoder with attention and input feeding
    """

    def _run_forward_pass(self, tgt, ctx, state, ctx_lengths):
        """
        Applies a multi-layer RNN with following attention to an target sequence.

        Args:
            tgt (batch_size, seq_len): tensor containing the features of the target sequence.
            ctx (batch_size, ctx_len, ctx_dim): tensor containing context.
            state (RNNDecoderState): decoder initial state.
            ctx_lengths (LongTensor): [batch_size] containing context lengths.

        Returns: 
            outputs (batch_size, seq_len, hidden_dim): outputs of the decoder
            hidden (FloatTensor): tensor or tuple containing last hidden states.
        """

        # initial state
        if self.rnn_type.lower() == 'gru':
            hidden = state.hidden[0]
        elif self.rnn_type.lower() == 'lstm':
            hidden = state.hidden
        output = state.input_feed.squeeze(0)

        # embedding
        batch_size, seq_len = tgt.size()
        emb = self.embedding(tgt) # batch_size * seq_len * emb_dim

        outputs = []
        for emb_t in emb.split(1, dim=1):
            emb_t = emb_t.squeeze(1)
            emb_t = torch.cat([emb_t, output], 1)
            rnn_output, hidden = self.rnn(emb_t, hidden)

            # attention calculation
            attn_t, ctx_t = self.attn_func(rnn_output, ctx, ctx_lengths)  # attention

            output = torch.cat([ctx_t, rnn_output], 1)
            output = self.linear_out(output)
            if self.attn_type.lower() in ['general', 'dot']:
                output = self.tanh(output)
            output = self.dropout(output)

            # outputs
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)

        return outputs, hidden

    def _build_rnn(self, rnn_type, input_dim, hidden_dim, num_layers, dropout):
        if rnn_type.lower() == 'gru':
            return StackedGRU(num_layers, input_dim, hidden_dim, dropout)
        elif rnn_type.lower() == 'lstm':
            return StackedLSTM(num_layers, input_dim, hidden_dim, dropout)
        else:
            raise Exception('Unsupported RNN Type')
    
    @property
    def _input_dim(self):
        return self.embedding.emb_dim + self.hidden_dim


class DecoderState(object):
    """
    DecoderState is a base class for models, used during translation for storing translation states
    """

    def detach(self):
        """
        Detaches all Variables from the graph that created it, make it a leaf
        """
        for h in self._all:
            if h is not None:
                h.detach_()

    def beam_update(self, idx, position, beam_size):
        """ Update when beam advances. """
        for e in self._all:
            if e is None:
                continue
            a, br, d =  e.size()
            sentState = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sentState.data.copy_(
                    sentState.data.index_select(1, position)
            )

class RNNDecoderState(DecoderState):
    def __init__(self, context, hidden_size, rnn_state):
        """
        Args:
            context (Variable): batch_size * enc_len * rnn_size
            hidden_size (int): hidden size
            rnn_state (Variable): layers * batch_size * rnn_size
            This should be transformed from enc_states in the decoder initilization.
            input_feed (Variable):  1 * batch_size * rnn_size, output from last layer of the decoder.
        """
        if not isinstance(rnn_state, tuple):
            self.hidden = (rnn_state,)
        else:
            self.hidden = rnn_state

        batch_size = context.size(0)
        h_size = (batch_size, hidden_size)
        self.input_feed = Variable(context.data.new(*h_size).zero_(), requires_grad=False).unsqueeze(0)

    @property
    def _all(self):
        return self.hidden + (self.input_feed,)

    def update_state(self, rnn_state, input_feed):
        if not isinstance(rnn_state, tuple):
            self.hidden = (rnn_state, )
        else:
            self.hidden = rnn_state
        self.input_feed = input_feed

    def repeat_beam_size_times(self, beam_size):
        """ Repeat beam_size times along batch dimensions. """
        vars = [Variable(e.data.repeat(1, beam_size, 1), volatile=True)
                for e in self._all if e is not None]
        self.hidden = tuple(vars[:-1])
        self.input_feed = vars[-1]

