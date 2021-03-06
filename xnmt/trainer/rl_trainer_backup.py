# -*- coding: utf-8 -*-
import os, sys
import logging
import math, random

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu

import torch
import torch.nn as nn

from xnmt.utils import drop_chkpt, load_chkpt, make_logger, Statistics, sequence_mask
from xnmt.io import Constants
from xnmt.trainer.rl_criterion import RLCriterion

class RLTrainer(object):
    """
    Class that controls the training process.

    Integrated with reinforcement learning mechanism. Reference paper:
        * SEQUENCE LEVEL TRAINING WITH RECURRENT NEURAL NETWORKS

    Args:
        
    """

    def __init__(self, model, reward_type, ce_criterion, optimizer, print_every, cuda=True):
        self.cuda = cuda
        self.model = model
        self.train_criterion = RLCriterion(reward_type, ce_criterion)
        self.valid_criterion = ce_criterion
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

    def train_on_batch(self, enc_data, enc_lengths, dec_data):
        # data initialization
        if self.cuda:
            enc_data, dec_data = enc_data.cuda(), dec_data.cuda()
            enc_lengths = enc_lengths.cuda()
        dec_inputs = dec_data[:, :-1]
            
        #target = dec_data[:, 1:].contiguous().view(-1)
        target = dec_data[:, 1:].contiguous()

        # encoder calculation
        enc_outputs, enc_hiddens = self.model.encoder(enc_data, enc_lengths)
        state = self.model.decoder.init_states(enc_outputs, enc_hiddens)
        
        # prepare sample size
        batch_size = enc_data.size(0) * self.sample_size
        target = target.repeat(self.sample_size, 1)
        ctx = enc_outputs.repeat(self.sample_size, 1, 1)
        ctx_lengths = enc_lengths.repeat(self.sample_size)
        state.repeat_beam_size_times(self.sample_size)

        # sample
        y_0 = torch.LongTensor([Constants.BOS for _ in range(batch_size)])
        y_0 = y_0.view(-1, 1)
        if self.cuda:
            y_0 = y_0.cuda()
        y_t = y_0
        # samples
        samples = []
        log_probs = []
        baselines = []

        sample_length = target.size(1)
        lengths = torch.ones(batch_size).type_as(y_t)
        before_eos = torch.ones(batch_size).type_as(y_t).byte()

        sample_length = min(sample_length, self.max_sample_length)
        t = 0
        while t < sample_length:
            t += 1
            
            # decoder step forward
            outputs, state = self.model.decoder(y_t, ctx, 
                    state, ctx_lengths=ctx_lengths)
            log_prob_t = self.model.generator(outputs)  # batch_size * vocab_size
            log_probs.append(log_prob_t)

            # sample next step inputs
            prob_t = torch.exp(log_prob_t)
            y_t = prob_t.multinomial(1)
            y_t = y_t.detach()  
            samples.append(y_t)  # batch_size * 1

            # eos judgement
            before_eos = torch.ne(y_t.view(-1), Constants.EOS) & before_eos
            lengths += before_eos.long()
            
            # baseline calculation
            b_t_h = state.hidden[0].detach()[-1,:,:].squeeze(0)
            #b_t_h = state.hidden[0][-1,:,:].squeeze(0)
            b_t = self.model.baseline(b_t_h)
            baselines.append(b_t)
            
            if torch.eq(y_t, Constants.EOS).type(torch.LongTensor).sum().item() == batch_size:
                break

        log_probs = torch.stack(log_probs, dim=1)
        samples = torch.cat(samples, dim=1)
        baseline = torch.cat(baselines, dim=1)
        # post processing
        loss, reward = self.train_criterion(log_probs, target, samples, lengths, baseline)
        #loss, reward = self.train_criterion(log_probs, target, samples, lengths)
        return loss, reward

    def train_on_epoch(self, data_iter, epoch):

        self.logger.info("Epoch {:02} begins training .......................".format(epoch))
        self.model.train()
        total_reward = 0.
        stats = Statistics()

        for (i, (enc_data, enc_lengths, dec_data, _)) in enumerate(data_iter):
            
            loss, reward = self.train_on_batch(enc_data, enc_lengths, dec_data)
            
            total_reward += reward
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (i > 0) and (i % self.print_every == 0):
                self.logger.info("Epoch {:02}, {:05}/{:05}; Reward: {:6.2f}; {:6.0f}s elapsed".format(
                     epoch, i, data_iter.batches, total_reward / (i+1) , stats.elapsed_time()
                    ))

        self.logger.info("Epoch {:02}, Reward: {:6.2f}; {:6.0f}s elapsed".format(
                epoch,  total_reward / data_iter.batches, stats.elapsed_time()
            ))
        return total_reward / data_iter.batches 

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
            loss = self.valid_criterion(probs, target)
            
            # statistics
            loss_data = torch.sum(loss).data.clone()
            batch_stat = self.get_stats(loss_data, probs.data, target.data)
            stats.update(batch_stat)

        self.logger.info("Epoch {:02}, accu: {:6.2f}; ppl: {:6.2f}; {:6.0f}s elapsed".format(
                epoch, stats.accuracy(), stats.ppl(), stats.elapsed_time()
            ))
        return stats.accuracy(), stats.ppl()

    def epoch_step(self, ppl, epoch):
        self.optimizer.update_learning_rate(ppl, epoch)

    def load_chkpt(self, chkpt, model, optimizer=None, use_gpu=True):
        
        """Consideration: for rl trainer, since we have already change the optimized criterion,
        we can use a totally new optimizer instead of the old optimizer
        """
        
        print('RL Specific, Load the checkpoint from {}'.format(chkpt))
        print("RL Specific, Use gpu is: {}".format(use_gpu))

        chkpt = torch.load(chkpt,
            map_location = lambda storage, loc: storage)
        epoch = chkpt['epoch']
        model.load_state_dict(chkpt['model'], strict=False)

        optimizer.set_parameters(model.named_parameters())

        return epoch, model, optimizer

    def train(self, train_data, epochs, valid_data, sample_size=10, 
            sample_length=20, resume_chkpt=None):
        """
        Args:
            train_data (iterator): train data iterator
            epochs (int): total epochs of training
            valid_data (iterator): valid data iterator
            sample_size (int): sample size of each sentence in every batch
            sample_length (int): max sample length of decoder
            resume_chkpt (str): resume checkpoint path
        """
        if resume_chkpt is not None:
            
            self.start_epoch, self.model, self.optimizer = \
                    self.load_chkpt(resume_chkpt, self.model, self.optimizer, self.cuda)
            self.start_epoch += 1

        self.sample_size = sample_size
        self.max_sample_length = sample_length

        for epoch in range(self.start_epoch, epochs+1):
            _ = self.train_on_epoch(train_data, epoch)
            acc, ppl = self.eval_on_epoch(valid_data, epoch)
            self.epoch_step(ppl, epoch)
            drop_chkpt(epoch, self.model, self.optimizer, acc, ppl)

