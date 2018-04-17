# -*- coding:utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class Embedding(nn.Module):
    r"""
    Applies a embedding layer to an input sequence.

    Args:
        vocab_size (int): the size of the vocab
        emb_dim (int): the dimensinality of each embedding vector
        padding_idx (int): pad the output with zeros whenever it encounters the index.
    """

    def __init__(self, vocab_size, emb_dim, padding_idx):

        super(Embedding, self).__init__()
        self.emb_dim = emb_dim
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=padding_idx, sparse=False)
        self.init_params()

    def forward(self, input):
        """
        Applies a embedding layer to an input sequence.

        Args:
            input (batch, seq_len): tensor containing the features of the input sequence.

        Returns: output, hidden
            - **output** (batch, seq_len, emb_dim): variable containing embedding of the input sequence
        """
        emb = self.embedding(input)
        return emb

    def init_params(self):
        for name, param in self.named_parameters():
            if name.endswith('weight'):
                nn.init.normal(param, 0, 0.01)
            elif name.endswith('bias'):
                nn.init.constant(param, 0)
            else:
                raise Exception('Wrong parameters')


def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
            if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)
    ])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


