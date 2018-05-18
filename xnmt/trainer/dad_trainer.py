# -*- coding: utf-8 -*-
import os, sys
import logging
import math, random
import sympy

import torch
import torch.nn as nn

from xnmt.utils import drop_chkpt, load_chkpt, make_logger, Statistics
from xnmt.io import Constants

class DADTrainer(object):
    """
    Class that controls the training process.

    Integrated with scheduled sampling mechanism. Reference paper:
        * Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks
        * SEQUENCE LEVEL TRAINING WITH RECURRENT NEURAL NETWORKS

    Args:
        
    """

    def __init__(self, model, criterion, optimizer, print_every, cuda=True):
        self.cuda = cuda
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.print_every = print_every
        self.start_epoch = 1
        self.logger = make_logger('log.train')

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
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        return Statistics(loss.item(), non_padding.sum().item(), num_correct)

    # sample rate calculation: 
    def get_sample_func(self, epochs, iterations, rate=0.5):
        """Sample rate calculation
        
        Inverse sigmoid decay from paper:
            * Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks

        Expect: Rate=rate when reaching last iteration in last_epoch.

        """
        x = sympy.symbols('x')
        #k = sympy.solve(x * sympy.ln(x) - epochs * iterations, x)
        k = sympy.solve(x * sympy.ln(x / rate - x) - epochs * iterations, x)
        k = float(k[0])

        def f(epoch, iteration):
            rate =  k / (k + math.exp(epoch * iteration / k))
            return rate

        return f

    def train_on_epoch(self, data_iter, epoch):

        self.logger.info("Epoch {:02} begins training .......................".format(epoch))
        self.model.train()
        stats = Statistics()

        for (i, (enc_data, enc_lengths, dec_data, _)) in enumerate(data_iter):
            
            # data initialization
            if self.cuda:
                enc_data, dec_data = enc_data.cuda(), dec_data.cuda()
                enc_lengths = enc_lengths.cuda()
            dec_inputs = dec_data[:, :-1]
            
            #target = dec_data[:, 1:].contiguous().view(-1)
            target = dec_data[:, 1:].contiguous()
            flatten_target = target.view(-1)

            # model calculation
            enc_outputs, enc_hiddens = self.model.encoder(enc_data, enc_lengths)
            state = self.model.decoder.init_states(enc_outputs, enc_hiddens)

            loss = 0.
            target_length = dec_inputs.size(1)
            sample_rate = self.schedule_func(epoch, i)

            probs = []
            for di in range(target_length):
                if di == 0 or random.random() < sample_rate:
                    tgt = dec_inputs[:, di:di+1]
                else:
                    tgt = sample_inputs
                dec_outputs, state = self.model.decoder(tgt, enc_outputs, state, ctx_lengths=enc_lengths)
                dec_probs = self.model.generator(dec_outputs)
                probs.append(dec_probs)

                loss += self.criterion(dec_probs, target[:, di])

                topv, topi = dec_probs.data.topk(1)
                sample_inputs = topi
            
            # statistics
            probs = torch.stack(probs, 1).view(flatten_target.size(0), -1)
            loss_data = loss.data.clone()
            batch_stat = self.get_stats(loss_data, probs.data, flatten_target.data)
            stats.update(batch_stat)
            
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (i > 0) and (i % self.print_every == 0):
                self.logger.info("Epoch {:02}, {:05}/{:05}; accu: {:6.2f}; ppl: {:6.2f}; {:6.0f}s elapsed".format(
                     epoch, i, data_iter.batches, stats.accuracy(), stats.ppl(), stats.elapsed_time()
                    ))

        self.logger.info("Epoch {:02}, accu: {:6.2f}; ppl: {:6.2f}; {:6.0f}s elapsed".format(
                epoch, stats.accuracy(), stats.ppl(), stats.elapsed_time()
            ))
        return stats.accuracy(), stats.ppl()

    def eval_on_epoch(self, data_iter, epoch):

        self.logger.info("Epoch {:02} begins validation .......................".format(epoch))
        self.model.eval()
        stats = Statistics()

        for (i, (enc_data, enc_lengths, dec_data, _)) in enumerate(data_iter):
            
            # data initialization
            if self.cuda:
                enc_data, dec_data = enc_data.cuda(), dec_data.cuda()
                enc_lengths = enc_lengths.cuda()
            dec_inputs = dec_data[:, :-1]
            target = dec_data[:, 1:].contiguous().view(-1)
            
            # model calculation
            probs = self.model(enc_data, enc_lengths, dec_inputs)
            loss = self.criterion(probs, target)
            
            # statistics
            loss_data = loss.data.clone()
            batch_stat = self.get_stats(loss_data, probs.data, target.data)
            stats.update(batch_stat)

        self.logger.info("Epoch {:02}, accu: {:6.2f}; ppl: {:6.2f}; {:6.0f}s elapsed".format(
                epoch, stats.accuracy(), stats.ppl(), stats.elapsed_time()
            ))
        return stats.accuracy(), stats.ppl()

    def epoch_step(self, ppl, epoch):
        self.optimizer.update_learning_rate(ppl, epoch)

    def train(self, train_data, epochs, valid_data, final_rate=0.5, resume_chkpt=None):
        """
        Args:
            train_data (iterator): train data iterator
            epochs (int): total epochs of training
            valid_data (iterator): valid data iterator
            final_rate (float): final sample rate for ground truth
            resume_chkpt (str): resume checkpoint path
        """
        if resume_chkpt is not None:
            self.start_epoch, self.model, self.optimizer = \
                    load_chkpt(resume_chkpt, self.model, self.optimizer, self.cuda)
            self.start_epoch += 1

        self.schedule_func = self.get_sample_func(epochs, len(train_data), final_rate)

        for epoch in range(self.start_epoch, epochs+1):
            _, _ = self.train_on_epoch(train_data, epoch)
            acc, ppl = self.eval_on_epoch(valid_data, epoch)
            self.epoch_step(ppl, epoch)
            drop_chkpt(epoch, self.model, self.optimizer, acc, ppl)

