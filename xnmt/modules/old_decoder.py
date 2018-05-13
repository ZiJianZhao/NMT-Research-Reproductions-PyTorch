# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable

from xnmt.modules.embedding import Embedding
from xnmt.modules.attention import Attention
from xnmt.modules.stacked_rnn import StackedGRU, StackedLSTM
from xnmt.modules.decoder import RNNDecoderState
from xnmt.utils import aeq

class DecoderS2S(nn.Module):
    r"""
    Basic decoder for naive sequence-to-sequence model

    Args:
        rnn_type (str): type of RNN cell, one of [RNN, GRU, LSTM]
        num_layers (int): number of recurrent layers
        hidden_dim (int): the dimensionality of the hidden state `h`
        embedding (nn.Module): predefined embedding module
        dropout (float, optional): dropout

    Note: the layers of decoder should be same as that of encoder.
    """

    def __init__(self, rnn_type, num_layers, hidden_dim, embedding, dropout=0.):

        super(DecoderS2S, self).__init__()
        
        self.rnn_type = rnn_type
        if self.rnn_type not in ["GRU", "LSTM"]:
            raise Exception("Unsupported RNN Type in decoder")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers 
        self.embedding = embedding

        self.rnn = getattr(nn, rnn_type)(embedding.emb_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout)

        self.init_params()

    def forward(self, tgt, ctx, state, ctx_lengths=None):
        """
        Applies a multi-layer RNN to an target sequence.

        Args:
            tgt (batch_size, seq_len): tensor containing the features of the target sequence.
            ctx (batch_size, ctx_len, ctx_dim): tensor containing context
            state (RNNDecoderState): decoder initial hidden state.
            ctx_lengths (LongTensor): [batch_size] containing context lengths

        Note: You must first call init_states

        Returns: output, hidden
            - **output** (batch * seq_len, hidden_dim+emb_dim+ctx_dim): variable containing the encoded features of the input sequence
            - **hidden** tensor or tuple containing last hidden states.
        """
        if self.rnn_type.lower() == 'gru':
            hidden = state.hidden[0]
        elif self.rnn_type.lower() == 'lstm':
            hidden = state.hidden

        batch_size, seq_len = tgt.size()
        emb = self.embedding(tgt) # batch_size * seq_len * emb_dim
        outputs, hidden = self.rnn(emb, hidden)
        outputs = outputs.contiguous().view(batch_size*seq_len, -1)

        state.update_state(hidden, None)

        return outputs, state

    def init_states(self, context, enc_states):
        """
        Initialize the hidden states decoder. We only uses the backward last state instead of the concated bidirectional state.

        Args:
            context (Variable): batch_size * enc_len * (enc_directions * enc_size)
            enc_states (tensor or tuple): (layers* directions, batch_size, enc_hidden_dim),

        Returns:
            dec_states (RNNDecoderState): 


        Consideration: This part is for modifying the encoder states to initialize decoder states. Since it is closely related to the 
            structure of decoder, it is a function inside of the decoder instead of a separate module. 
        """
        
        if isinstance(enc_states, tuple):
            aeq(enc_states[0].size(2), self.hidden_dim)
        else:
            aeq(enc_states.size(2), self.hidden_dim)

        if isinstance(enc_states, tuple):  # the encoder is a LSTM
            if self.rnn_type == 'GRU':
                if enc_states[0].size(0) == self.num_layers:
                    return RNNDecoderState(context, self.hidden_dim, enc_states[0])
                else:
                    l = enc_states[0].size(0)
                    return RNNDecoderState(context, self.hidden_dim, enc_states[0][1:l:2])
            elif self.rnn_type == 'LSTM':
                if enc_states[0].size(0) == self.num_layers:
                    return RNNDecoderState(context, self.hidden_dim, enc_states)
                else:
                    l = enc_states[0].size(0)
                    hidden = (enc_states[0][1:l:2], enc_states[0][1:l:2])
                    return RNNDecoderState(context, self.hidden_dim, hidden)
        else:
            if self.rnn_type == 'GRU':
                if enc_states.size(0) == self.num_layers:
                    return RNNDecoderState(context, self.hidden_dim, enc_states)
                else:
                    l = enc_states.size(0)
                    hidden = enc_states[1:l:2]
                    return RNNDecoderState(context, self.hidden_dim, hidden)
            else:
                raise Exception("Unsupported structure: encoder-gru --> decoder-lstm")

    def init_params(self):
        """
        This initializaiton is especially for gru according to the paper:
            Neural Machine Translation by Jointly Learning to Align and Translate
        """
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.normal(param, 0, 0.01)
            elif 'weight_hh' in name:
                for i in range(0, param.data.size(0), self.hidden_dim):
                    nn.init.orthogonal(param.data[i:i+self.hidden_dim])
            elif 'bias' in name:
                nn.init.constant(param, 0)
            elif 'enc2dec' in name and 'weight' in name:
                nn.init.normal(param, 0, 0.01)

class DecoderRNNsearch(nn.Module):
    r"""
    Applies a multi-layer RNN to an input sequence. Following Structure of the paper:
        Neural Machine Translation by Jointly Learning to Align and Translate

    Note: only support single layer RNN. A lot of settings are specific to the paper.

    This module can only be compatible with a single-layer bidirectional encoder for replicate the paper above. 

    Args:
        rnn_type (str): type of RNN cell, one of [RNN, GRU, LSTM]
        attn_type (str): type of attention, one of [mlp, dot, general]
        hidden_dim (int): the dimensionality of the hidden state `h`
        embedding (nn.Module): predefined embedding module
        ctx_dim (int): the dimensionality of context vector
    """

    def __init__(self, rnn_type, attn_type, hidden_dim, embedding, ctx_dim):

        super(DecoderRNNsearch, self).__init__()
        
        self.rnn_type = rnn_type
        if self.rnn_type not in ["GRU", "LSTM"]:
            raise Exception("Unsupported RNN Type in decoder")
        self.hidden_dim = hidden_dim
        
        # define parameters for transform encoder states to decoder states for initialization
        # this is specific for the paper above
        # for other possible configurations, we just share the parameters
        self.enc2dec = nn.Linear(hidden_dim, hidden_dim)
        self.tanh = nn.Tanh()

        self.embedding = embedding
        self.attn_func = Attention(attn_type, ctx_dim, hidden_dim)
        # here need to use rnncell instead of rnn, since we need to explictly unroll
        self.rnn = getattr(nn, rnn_type+'Cell')(embedding.emb_dim + ctx_dim, hidden_dim)

        self.init_params()

    def forward(self, tgt, ctx, state, ctx_lengths=None):
        """
        Applies a multi-layer RNN to an target sequence.

        Args:
            tgt (batch_size, seq_len): tensor containing the features of the target sequence.
            ctx (batch_size, ctx_len, ctx_dim): tensor containing context
            state (RNNDecoderState): decoder initial hidden state
            ctx_lengths (LongTensor): [batch_size] containing context lengths

        Returns: output, hidden
            - **output** (batch * seq_len, hidden_dim+emb_dim+ctx_dim): variable containing the encoded features of the input sequence
            - **hidden** tensor or tuple containing last hidden states.
        """
        if self.rnn_type.lower() == 'gru':
            hidden = state.hidden[0].squeeze(0)
        elif self.rnn_type.lower() == 'lstm':
            hidden = tuple(h.squeeze(0) for h in state.hidden)

        batch_size, seq_len = tgt.size()
        emb = self.embedding(tgt) # batch_size * seq_len * emb_dim
        outputs = []
        for emb_t in emb.split(1, dim=1):
            emb_t = emb_t.squeeze(1)

            # attention calculation
            if self.rnn_type == 'GRU':
                attn_t, ctx_t = self.attn_func(hidden, ctx, ctx_lengths)  # attention
            elif self.rnn_type == 'LSTM':
                attn_t, ctx_t = self.attn_func(hidden[0], ctx, ctx_lengths)  # attention

            # RNN operation
            input_t = torch.cat([emb_t, ctx_t], 1)
            if self.rnn_type == 'GRU':
                output = torch.cat([hidden, emb_t, ctx_t], 1)
            else:
                output = torch.cat([hidden[0], emb_t, ctx_t], 1)
            hidden = self.rnn(input_t, hidden)

            # outputs
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.contiguous().view(batch_size*seq_len, -1)


        if self.rnn_type.lower() == 'gru':
            hidden = hidden.unsqueeze(0)
        elif self.rnn_type.lower() == 'lstm':
            hidden = tuple(h.unsqueeze(0) for h in hidden)
        state.update_state(hidden, None)

        return outputs, state

    def init_states(self, context, enc_states):
        """
        Initialize the hidden states decoder. The paper only uses the backward last state instead of the concated bidirectional state

        Args:
            context (Variable): batch_size * enc_len * (enc_directions * enc_size)
            enc_states (tensor or tuple): (2, batch_size, enc_hidden_dim), 2 * enc_hidden_dim == self.ctx_dim

        Returns:
            dec_states (RNNDecoderState): 

        Note: ctx_dim / hidden_dim == 2

        Consideration: This part is for modifying the encoder states to initialize decoder states. Since it is closely related to the 
            structure of decoder, it is a function inside of the decoder instead of a separate module. 
        """
        
        if isinstance(enc_states, tuple):
            aeq(enc_states[0].size(2), self.hidden_dim)
        else:
            aeq(enc_states.size(2), self.hidden_dim)

        if isinstance(enc_states, tuple):  # the encoder is a LSTM
            enc_states = (enc_states[0][1], enc_states[1][1]) # batch_size * hidden_dim, only save the backward state
            if self.rnn_type == 'GRU':
                hidden = self.tanh(self.enc2dec(enc_states[0])).unsqueeze(0)
                return RNNDecoderState(context, self.hidden_dim, hidden)
            elif self.rnn_type == 'LSTM':
                hidden = (self.tanh(self.enc2dec(enc_states[0])).unsqueeze(0), 
                    self.tanh(self.enc2dec(enc_states[1])).unsqueeze(0)
                )
                return RNNDecoderState(context, self.hidden_dim, hidden)
        else:
            enc_states = enc_states[1]
            if self.rnn_type == 'GRU':
                hidden = self.tanh(self.enc2dec(enc_states)).unsqueeze(0)
                return RNNDecoderState(context, self.hidden_dim, hidden)
            else:
                raise Exception("Unsupported structure: encoder-gru --> decoder-lstm")

    def init_params(self):
        """
        This initializaiton is especially for gru according to the paper:
            Neural Machine Translation by Jointly Learning to Align and Translate
        """
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.normal(param, 0, 0.01)
            elif 'weight_hh' in name:
                for i in range(0, param.data.size(0), self.hidden_dim):
                    nn.init.orthogonal(param.data[i:i+self.hidden_dim])
            elif 'bias' in name:
                nn.init.constant(param, 0)
            elif 'enc2dec' in name and 'weight' in name:
                nn.init.normal(param, 0, 0.01)

