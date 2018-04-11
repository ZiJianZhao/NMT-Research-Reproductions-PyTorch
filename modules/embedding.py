# -*- coding:utf-8 -*-
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

if __name__ == '__main__':
    model = Embedding(10, 10, 0)
    input = Variable(torch.LongTensor([[1,2,3],[4,5,6]]))
    emb = model(input)
    print(emb)

