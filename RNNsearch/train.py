# -*- codind: utf-8 -*-

import os, sys, random, argparse
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

sys.path.append('../')
from xnmt.io.dataloader import DataLoader, PreNDataLoader
from xnmt.io import Constants
from xnmt.trainer import Trainer
from xnmt.optim import Optim

from model import make_model

def train_opts(parser):
    # Data options

    parser.add_argument('-data', required=True,
                       help="Path prefix to the train.pt and valid.pt")
    parser.add_argument('-save_model', default='chkpts/ldc',
                       help="Saved model filename")
    parser.add_argument('-gpuid', default=0, type=int,
                       help="gpu id to run")
    parser.add_argument('-batch_size', default=80, type=int,
                       help="batch size")
    parser.add_argument('-epochs', default=20, type=int,
                       help="epochs")
    parser.add_argument('-print_every', default=100, type=int,
                       help="print every batches")
    parser.add_argument('-seed', default=3435,
                       help='random seed')

def parse_args():
    parser = argparse.ArgumentParser(
            description='Train Options',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    train_opts(parser)
    opt = parser.parse_args()

    # fix random seed
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    
    return opt

def train():

    # Initialization
    opt = parse_args()

    if opt.gpuid >= 0:
        torch.cuda.set_device(opt.gpuid)

    # Data iterator definition
    print("Loading data ....")
    train_data = torch.load(opt.data+'.train.pt')
    print("Train data: {}".format(len(train_data)))
    train_iter = PreNDataLoader(train_data, opt.batch_size, epoch_shuffle=True)

    valid_data = torch.load(opt.data+'.valid.pt')
    print("Valid data: {}".format(len(valid_data)))
    valid_iter = PreNDataLoader(valid_data, opt.batch_size, epoch_shuffle=False)
    src_word2idx = torch.load(opt.data+'.vocab.pt')['src']
    tgt_word2idx = torch.load(opt.data+'.vocab.pt')['tgt']
    print("src_vocab: {}, tgt_vocab: {}".format(len(src_word2idx), len(tgt_word2idx)))

    # Model definition
    model = make_model(len(src_word2idx), len(tgt_word2idx))
    if opt.gpuid >= 0:
        model = model.cuda()

    optimizer = Optim('Adadelta', 1.0, max_grad_norm=5)
    optimizer.set_parameters(model.named_parameters())

    weight = torch.ones(len(tgt_word2idx))
    weight[Constants.PAD] = 0 
    criterion = nn.NLLLoss(weight, size_average=False)
    if opt.gpuid >= 0:
        criterion = criterion.cuda()

    trainer = Trainer(model, criterion, optimizer, opt.print_every)

    trainer.train(train_iter, opt.epochs, valid_iter)

if __name__ == '__main__':
    train()
