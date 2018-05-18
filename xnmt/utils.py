# -*- coding: utf-8 -*-
import os, datetime
import logging
import time, math

import torch
import torch.nn as nn


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
            "Not all arguments have the same value: " + str(args)

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    Borrow from OpenNMT-py: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Utils.py

    Args:
        lengths (LongTensor): containing sequence lengths
        max_len (int): maximum padded length
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

class Statistics(object):
    """
    Accumulator for loss statistics:
        * accuracy
        * perplexity
        * elapsed time
    """
    def __init__(self, loss=0., n_words=0., n_correct=0.):
        self.loss = loss
        self.n_words = n_words 
        self.n_correct = n_correct
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def ppl(self):
        return math.exp(min(self.loss / self.n_words, 100))

    def elapsed_time(self):
        return time.time() - self.start_time


def drop_chkpt(epoch, model, optimizer, accu=None, ppl=None):
    """
    Drop a checkpoint, if have ppl and accu, display them in the name
    """
    i = datetime.datetime.now()
    #chkpt_dir = "Y_{}_M_{}_D_{}_chkpts".format(i.year, i.month, i.day)
    chkpt_dir = 'chkpts'
    if not os.path.exists(chkpt_dir):
        os.mkdir(chkpt_dir)
    
    real_model = model.module if isinstance(model, nn.DataParallel) else model

    chkpt = {
        'epoch': epoch,
        'model': real_model.state_dict(),
        'optimizer': optimizer
    }
    if ppl is None or accu is None:
        name = chkpt_dir + '/' + 'chkpt_{:02}.pt'.format(epoch)
    else:
        name = chkpt_dir + '/' + 'chkpt_{:02}_accu_{:.2f}_ppl_{:.2f}.pt'.format(epoch, accu, ppl)
    torch.save(chkpt, name)
    print("Drop a checkpoint at {}".format(name))

def load_chkpt(chkpt, model, optimizer=None, use_gpu=True):

    print('Load the checkpoint from {}'.format(chkpt))
    print("Use gpu is: {}".format(use_gpu))

    chkpt = torch.load(chkpt,
            map_location = lambda storage, loc: storage)
    epoch = chkpt['epoch']
    model.load_state_dict(chkpt['model'])

    if optimizer is not None:
        
        optimizer = chkpt['optimizer']
        saved_optimizer_state_dict = optimizer.optimizer.state_dict()
        optimizer.set_parameters(model.named_parameters())

        optimizer.optimizer.load_state_dict(saved_optimizer_state_dict)

        if use_gpu:
            for state in optimizer.optimizer.state.values():
                for k,v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
    
    if optimizer is not None:
        return epoch, model, optimizer
    else:
        return epoch, model

def make_logger(filename):
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(filename)
    handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def tally_parameters(model, logger=None):
    if logger is not None:
        func = logger.info
    else:
        func = print
    n_params = sum([p.nelement() for p in model.parameters()])
    func('* number of parameters: {}'.format(n_params))
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    func('encoder: {}'.format(enc))
    func('decoder: {}'.format(dec))

