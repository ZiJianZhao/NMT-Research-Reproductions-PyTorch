# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import aeq

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
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, src, src_lengths, tgt):
        # encoder
        enc_outputs, enc_hiddens = self.encoder(src, src_lengths)

        # decoder
        dec_outputs, dec_hiddens = self.decoder(tgt, enc_outputs, ctx_lengths=src_lengths, enc_states=enc_hiddens)

        # generator
        probs = self.generator(dec_outputs)

        return probs



