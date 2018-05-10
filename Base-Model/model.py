import os, sys

import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('../')
from xnmt.modules import Embedding, EncoderRNN, StdRNNDecoder, NMTModel, InputFeedRNNDecoder
from xnmt.io import Constants

def make_model(src_vocab_size, tgt_vocab_size, dropout=0.3):
    
    # encoder
    enc_emb = Embedding(src_vocab_size, 500, Constants.PAD, dropout)
    encoder = EncoderRNN('LSTM', True, 2, 500, enc_emb, dropout)

    # decoder
    dec_emb = Embedding(tgt_vocab_size, 500, Constants.PAD, dropout)
    #decoder = InputFeedRNNDecoder('LSTM', 2, 1000, 'dot', 1000, dec_emb, True, dropout)
    decoder = StdRNNDecoder('LSTM', 2, 1000, 'dot', 1000, dec_emb, True, dropout)
    
    # generator
    generator = nn.Sequential(
                nn.Linear(1000, tgt_vocab_size),
                nn.LogSoftmax(dim=1)
            )

    model = NMTModel(encoder, decoder, generator)

    for p in model.parameters():
        p.data.uniform_(-0.1, 0.1)

    return model

