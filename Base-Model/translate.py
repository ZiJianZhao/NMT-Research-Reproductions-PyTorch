# -*- codind: utf-8 -*-

import os, sys, random, argparse
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

sys.path.append('../')
from xnmt.io.dataloader import DataLoader, PreNDataLoader
from xnmt.io import Constants
from xnmt.trainer import Trainer
from xnmt.translate.translator import Translator

from model import make_model

def translate_opts(parser):
    # Data options

    parser.add_argument('-test_src', required=True,
                       help="filename of the source text to be translated")
    parser.add_argument('-vocab', required=True,
                       help="filename of the vocab")
    parser.add_argument('-chkpt', required=True,
                       help="Load model filename")
    parser.add_argument('-out_file', default='out.txt',
                       help="filename of translated text")
    parser.add_argument('-gpuid', default=0, type=int,
                       help="gpu id to run")
    parser.add_argument('-batch_size', default=1, type=int,
                       help="batch size")
    parser.add_argument('-beam_size', default=10, type=int,
                       help="beam size")
    parser.add_argument('-n_best', default=1, type=int,
                       help="n_best")
    parser.add_argument('-max_length', default=80, type=int,
                       help="max length")
    parser.add_argument('-seed', default=3435,
                       help='random seed')

def parse_args():
    parser = argparse.ArgumentParser(
            description='Translate Options',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    translate_opts(parser)
    opt = parser.parse_args()

    # fix random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)

    if opt.gpuid >= 0:
        torch.cuda.set_device(opt.gpuid)
        opt.cuda = True
    else:
        opt.cuda = False
    
    return opt

def translate():

    # Initialization
    opt = parse_args()

    src_word2idx = torch.load(opt.vocab)['src']
    tgt_word2idx = torch.load(opt.vocab)['tgt']
    print("src_vocab: {}, tgt_vocab: {}".format(len(src_word2idx), len(tgt_word2idx)))

    # Model definition
    model = make_model(len(src_word2idx), len(tgt_word2idx))

    translator = Translator(model, opt.beam_size, opt.n_best, opt.max_length)
    translator.translate(opt)

if __name__ == '__main__':
    translate()
