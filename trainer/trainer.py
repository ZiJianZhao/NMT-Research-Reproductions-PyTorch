# -*- coding: utf-8 -*-

import logging

import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('../')
from utils import drop_chkpt, load_chkpt, make_logger, Statistics
from dataset import Constants

class Trainer(object):
    """
    Class that controls the training process

    Args:
        
    """

    def __init__(self, model, criterion, optimizer, print_every, cuda=True):
        self.cuda = cuda
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.print_every = print_every
        self.start_epoch = 1
        self.logger = make_logger('train.log')

    def get_stats(self, loss, probs, target):
        """
        Args:
            loss (FloatTensor): the loss computed by the loss criterion.
            probs (FloatTensor): the generated probs of the model.
            target (LongTensor): true targets

        Returns:
            stats (Statistics): statistics for this batch
        """
        pred = probs.max(1)[1] # predicted targets
        non_padding = target.ne(Constants.PAD)
        num_correct = pred.eq(target).masked_select(non_padding).sum()
        return Statistics(loss[0], non.padding.sum(), num_correct)

    def train_on_epoch(data_iter, epoch, is_train):
        if is_train:
            self.logger.info("Epoch {:02} begins training .......................".format(epoch))
            model.train()
        else:
            self.logger.info("Epoch {:02} begins validation .......................".format(epoch))
            model.eval()
        stats = Statistics()
        for (i, (enc_data, enc_lengths, dec_data, _)) in enumerate(data_iter):
            
            # data initialization
            enc_data = Variable(enc_data, volatile=not is_train)
            dec_data = Variable(dec_data, volatile=not is_train)
            if self.cuda:
                enc_data, dec_data = enc_data.cuda(), dec_data()
            dec_inputs = dec_data[:, :-1]
            target = dec_data[:, 1:].contiguous().view(-1)
            
            # model calculation
            probs = self.model(enc_data, enc_lengths, dec_inputs)
            loss = self.criterion(probs, target)
            
            # statistics
            loss_data = loss.data.clone()
            batch_stat = self.get_stats(loss_data, probs.data, target.data)
            stats.update(batch_stat)
            
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            if is_train and (i > 0) and (i % self.print_every == 0):
                self.logger.info("Epoch {:02}, {:05}/{:05}; accu: {:6.2f}; ppl: {:6.2f}; {:6.0f}s elapsed".format(
                     epoch, i, data_iter.batches, stats.accuracy(), stats.ppl(), stats.elapsed_time()
                    ))

        if is_train:
            drop_chkpt(epoch, self.model, self.optimizer, stats.accuracy(), stats.ppl())

        self.logger.info("Epoch {:02}, accu: {:6.2f}; ppl: {:6.2f}; {:6.0f}s elapsed".format(
                epoch, stats.accuracy(), stats.ppl(), stats.elapsed_time()
            ))


    def train(train_data, epochs, valid_data, resume_chkpt=None):
        """
        Args:
            train_data (iterator): train data iterator
            epochs (int): total epochs of training
            valid_data (iterator): valid data iterator
            resume_chkpt (str): resume checkpoint path
        """
        if resume_chkpt is not None:
            self.start_epoch, self.model, self.optimizer = \
                    load_chkpt(resume_chkpt, self.model, self.optimizer)
            self.start_epoch += 1

        for epoch in range(self.start_epoch, epochs+1):
            self.train_on_epoch(train_data, epoch, True)
            self.train_on_epoch(valid_data, epoch, False)

if __name__ == '__main__':





