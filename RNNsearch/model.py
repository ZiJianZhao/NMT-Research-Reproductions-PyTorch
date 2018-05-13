import os, sys

import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('../')
from xnmt.modules import Embedding, EncoderRNN, DecoderRNNsearch, NMTModel, Maxout
from xnmt.io import Constants

def make_model(src_vocab_size, tgt_vocab_size):
    
    # encoder
    enc_emb = Embedding(src_vocab_size, 620, Constants.PAD)
    encoder = EncoderRNN('GRU', True, 1, 1000, enc_emb)

    # decoder
    dec_emb = Embedding(tgt_vocab_size, 620, Constants.PAD)
    decoder = DecoderRNNsearch('GRU', 'mlp', 1000, dec_emb, 2000)
    #decoder = DecoderS2S('GRU', 1, 100, dec_emb)
    
    # generator
    generator = nn.Sequential(
                Maxout(3620, 500, 2),
                nn.Linear(500, tgt_vocab_size),
                nn.LogSoftmax(dim=1)
            )
    return NMTModel(encoder, decoder, generator)

