# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable

from xnmt.utils import aeq

class NMTModel(nn.Module):
    """
    Define a general interface for encoder-decoder framework.

    * Benefits:
        * help to define a general interface for training and testing (decoding)

    × Modules:
        * encoder: encode the src features, return sentence-level series representation 'output' and 
            last hidden state 'hidden'.
        * decoder: encode the tgt features with the representation from encoder.
        * generator: output layer (usually softmax), operate on the representation from decoder to
            output probability on predicted labels

    Note: the modules should be compatible.
    """
    def __init__(self, encoder, decoder, generator):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, src, src_lengths, tgt):
        # ----- General Interface Considerations -----------------------
        
        # Encoder
        # Encoder is just a RNN for now.
        # And we always use nn.GRU and nn.LSTM for implementations
        # enc_outputs: (batch_size, seq_len, hidden_dim * num_directions)
        # enc_hiddens: (num_layers * num_directions, batch_size, hidden_dim)
        enc_outputs, enc_hiddens = self.encoder(src, src_lengths)

        # Decoder
        # Decoder is always a uni-directional RNN.
        # And we always use nn.GRUCell and nn.LSTMCell for implementation (Even for simple seq2seq)
        # Note: The purpose is to define a general translator even in cost of speed
        dec_outputs, dec_hiddens = self.decoder(tgt, enc_outputs, ctx_lengths=src_lengths, enc_states=enc_hiddens)

        # generator
        probs = self.generator(dec_outputs)

        return probs



