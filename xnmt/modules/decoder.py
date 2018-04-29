# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable

from xnmt.modules.embedding import Embedding
from xnmt.modules.attention import Attention
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

    def forward(self, tgt, ctx, h0=None, ctx_lengths=None, enc_states=None):
        """
        Applies a multi-layer RNN to an target sequence.

        Args:
            tgt (batch_size, seq_len): tensor containing the features of the target sequence.
            ctx (batch_size, ctx_len, ctx_dim): tensor containing context
            h0: tensor or tuple containing the initial hidden state.
            ctx_lengths (LongTensor): [batch_size] containing context lengths
            enc_states (tensor or tuple): (2, batch_size, enc_hidden_dim), 2 * enc_hidden_dim == self.ctx_dim, 
                used to compute the initial hidden states of decoder.

        Note: enc_states is needed when training the model and in the first step of decoding in test,
        h0 is for the follow-up steps of decoding in test. 

        Returns: output, hidden
            - **output** (batch * seq_len, hidden_dim+emb_dim+ctx_dim): variable containing the encoded features of the input sequence
            - **hidden** tensor or tuple containing last hidden states.
        """
        if h0 is not None:
            hidden = h0
        else:
            assert enc_states is not None, "Wrong, no way to initial the hidden states of decoder"
            hidden = self.init_states(enc_states)

        batch_size, seq_len = tgt.size()
        emb = self.embedding(tgt) # batch_size * seq_len * emb_dim
        outputs, hidden = self.rnn(emb, hidden)
        outputs = outputs.contiguous().view(batch_size*seq_len, -1)
        return outputs, hidden

    def init_states(self, enc_states):
        """
        Initialize the hidden states decoder. We only uses the backward last state instead of the concated bidirectional state.

        Args:
            enc_states (tensor or tuple): (layers* directions, batch_size, enc_hidden_dim),

        Returns:
            dec_states (tensor or tuple): (layers, batch_size, dec_hidden_dim)


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
                    return enc_states[0]
                else:
                    l = enc_states[0].size(0)
                    return enc_states[0][1:l:2]
            elif self.rnn_type == 'LSTM':
                if enc_states[0].size(0) == self.num_layers:
                    return enc_states
                else:
                    l = enc_states[0].size(0)
                    return (enc_states[0][1:l:2], enc_states[0][1:l:2])
        else:
            if self.rnn_type == 'GRU':
                if enc_states.size(0) == self.num_layers:
                    return enc_states
                else:
                    l = enc_states.size(0)
                    return enc_states[1:l:2]
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

    def forward(self, tgt, ctx, h0=None, ctx_lengths=None, enc_states=None):
        """
        Applies a multi-layer RNN to an target sequence.

        Args:
            tgt (batch_size, seq_len): tensor containing the features of the target sequence.
            ctx (batch_size, ctx_len, ctx_dim): tensor containing context
            h0: tensor or tuple containing the initial hidden state.
            ctx_lengths (LongTensor): [batch_size] containing context lengths
            enc_states (tensor or tuple): (2, batch_size, enc_hidden_dim), 2 * enc_hidden_dim == self.ctx_dim, 
                used to compute the initial hidden states of decoder.

        Note: enc_states is needed when training the model and in the first step of decoding in test,
        h0 is for the follow-up steps of decoding in test. 

        Returns: output, hidden
            - **output** (batch * seq_len, hidden_dim+emb_dim+ctx_dim): variable containing the encoded features of the input sequence
            - **hidden** tensor or tuple containing last hidden states.
        """
        if h0 is not None:
            hidden = h0
        else:
            assert enc_states is not None, "Wrong, no way to initial the hidden states of decoder"
            hidden = self.init_states(enc_states)

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
        return outputs, hidden

    def init_states(self, enc_states):
        """
        Initialize the hidden states decoder. The paper only uses the backward last state instead of the concated bidirectional state

        Args:
            enc_states (tensor or tuple): (2, batch_size, enc_hidden_dim), 2 * enc_hidden_dim == self.ctx_dim

        Returns:
            dec_states (tensor or tuple): (batch_size, hidden_dim), since we use the RNNCell instead of RNN.

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
                return self.tanh(self.enc2dec(enc_states[0]))
            elif self.rnn_type == 'LSTM':
                return (self.tanh(self.enc2dec(enc_states[0])), 
                    self.tanh(self.enc2dec(enc_states[1]))
                )
        else:
            enc_states = enc_states[1]
            if self.rnn_type == 'GRU':
                return self.tanh(self.enc2dec(enc_states))
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


class DecoderStdRNN(nn.Module):
    r"""
    Follow the implementation of 'stdRNNDecoder' in OpenNMT-py 
    Reference: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Models.py
    Model overview: First RNN, then Attention
    Most probable paper:
        Effective Approaches to Attention-based Neural Machine Translation
    
    Note: the layers of decoder should be same as that of encoder.

    Args:
        rnn_type (str): type of RNN cell, one of [RNN, GRU, LSTM]
        num_layers (int): number of recurrent layers
        attn_type (str): type of attention, one of [mlp, dot, general]
        hidden_dim (int): the dimensionality of the hidden state `h`
        embedding (nn.Module): predefined embedding module
        ctx_dim (int): the dimensionality of context vector
        bidirectional_encoder (bool): whether the encoder is bidirectional
        dropout (float, optional): dropout
    """

    def __init__(self, rnn_type, num_layers, attn_type, hidden_dim, embedding, 
            ctx_dim, bidirectional_encoder=False, dropout=0.):

        super(DecoderStdRNN, self).__init__()
        
        self.rnn_type = rnn_type
        self.bidirectional_encoder = bidirectional_encoder
        if bidirectional_encoder:
            self.encoder_directions = 2
        else:
            self.encoder_directions = 1
        if self.rnn_type not in ["GRU", "LSTM"]:
            raise Exception("Unsupported RNN Type in decoder")
        self.hidden_dim = hidden_dim
        self.embedding = embedding
        self.attn_func = Attention(attn_type, ctx_dim, hidden_dim)
        self.rnn = getattr(nn, rnn_type)(embedding.emb_dim, hidden_dim, num_layers,
                batch_first=True, dropout=dropout)

        self.init_params()

    def forward(self, tgt, ctx, h0=None, ctx_lengths=None, enc_states=None):
        """
        Applies a multi-layer RNN with following attention to an target sequence.

        Args:
            tgt (batch_size, seq_len): tensor containing the features of the target sequence.
            ctx (batch_size, ctx_len, ctx_dim): tensor containing context
            h0: tensor or tuple containing the initial hidden state.
            ctx_lengths (LongTensor): [batch_size] containing context lengths
            enc_states (tensor or tuple): (2, batch_size, enc_hidden_dim), 2 * enc_hidden_dim == self.ctx_dim, 
                used to compute the initial hidden states of decoder.

        Note: enc_states is needed when training the model and in the first step of decoding in test,
        h0 is for the follow-up steps of decoding in test. 

        Returns: output, hidden
            - **output** (batch * seq_len, hidden_dim+emb_dim+ctx_dim): variable containing the encoded features of the input sequence
            - **hidden** tensor or tuple containing last hidden states.
        """
        if h0 is not None:
            hidden = h0
        else:
            assert enc_states is not None, "Wrong, no way to initial the hidden states of decoder"
            hidden = self.init_states(enc_states)

        batch_size, seq_len = tgt.size()
        emb = self.embedding(tgt) # batch_size * seq_len * emb_dim

        outputs, hidden = self.rnn(emb, hidden)

        attns, ctxs = self.attn_func(outputs, ctx, ctx_lengths)

        outputs = torch.cat([ctxs, outputs], 2)

        outputs = outputs.contiguous().view(batch_size*seq_len, -1)
    
        return outputs, hidden

    def init_states(self, enc_states):
        """
        Initialize the hidden states decoder. 
        Following the implementation choice in OpenNMT-py:
            - decoder_hidden_dim = encoder_num_directions * encoder_hidden_dim

        Args:
            enc_states (tensor or tuple): (layers* directions, batch_size, enc_hidden_dim),

        Returns:
            dec_states (tensor or tuple): (layers, batch_size, dec_hidden_dim)
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
                return fix_state(enc_states[0])
            elif self.rnn_type == 'LSTM':
                return tuple([fix_state(h) for h in enc_states])
        else:
            if self.rnn_type == 'GRU':
                return fix_state(enc_states)
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
