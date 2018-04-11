# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable

from embedding import Embedding
from attention import Attention

class DecoderRNNsearch(nn.Module):
    r"""
    Applies a multi-layer RNN to an input sequence. Following Structure of the paper:
        Neural Machine Translation by Jointly Learning to Align and Translate

    Note: only support single layer RNN.

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
        self.hidden_dim = hidden_dim
        self.embedding = embedding
        self.attn_func = Attention(attn_type, ctx_dim, hidden_dim)
        # here need to use rnncell instead of rnn, since we need to explictly unroll
        self.rnn = getattr(nn, rnn_type+'Cell')(embedding.emb_dim + ctx_dim, hidden_dim)
        self.init_params()

    def forward(self, tgt, ctx, h0, ctx_lengths=None):
        """
        Applies a multi-layer RNN to an target sequence.

        Args:
            tgt (batch_size, seq_len): tensor containing the features of the target sequence.
            ctx (batch_size, ctx_len, ctx_dim): tensor containing context
            h0: tensor or tuple containing the initial hidden state.
            ctx_lengths (LongTensor): [batch_size] containing context lengths

        Returns: output, hidden
            - **output** (batch, seq_len, hidden_dim): variable containing the encoded features of the input sequence
            - **hidden** tensor or tuple containing last hidden states.
        """
        hidden = h0 
        emb = self.embedding(tgt) # batch_size * seq_len * emb_dim
        outputs = []
        for emb_t in emb.split(1, dim=1):
            emb_t = emb_t.squeeze(1)

            # attention calculation
            if self.rnn_type == 'GRU':
                attn_t, ctx_t = self.attn_func(hidden, ctx, ctx_lengths)  # attention
            elif self.rnn_type == 'LSTM':
                attn_t, ctx_t = self.atten_func(hidden[0], ctx, ctx_lengths)  # attention
            else:
                raise Exception("Unsupported RNN Type in decoder forward")

            # RNN operation
            input_t = torch.cat([emb_t, ctx_t], 1)
            hidden = self.rnn(input_t, hidden)

            # outputs
            if self.rnn_type == 'GRU':
                outputs.append(hidden)
            elif self.rnn_type == 'LSTM':
                outputs.append(hidden[0])
            else:
                raise Exception("Unsupported RNN Type in decoder forward")
        outputs = torch.stack(outputs, dim=1)
        return outputs, hidden

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

if __name__ == '__main__':
    torch.manual_seed(2)
    emb = Embedding(60, 6, 0)
    tgt = Variable(torch.LongTensor([[1,2,3],[4,5,6]]))
    print(tgt.size())
    ctx = Variable(torch.randn(2, 5, 20))
    h0 = Variable(torch.randn(2, 10))
    decoder = DecoderRNNsearch('GRU', 'mlp', 10, emb, 20)
    outputs, hidden = decoder(tgt, ctx, h0)
    print(outputs.size(), hidden.size())
