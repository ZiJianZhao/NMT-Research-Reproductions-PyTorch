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

class MixerTrainer(object):
    """
    Class that controls the training process.

    Integrated with reinforcement learning mechanism. Reference paper:
        * SEQUENCE LEVEL TRAINING WITH RECURRENT NEURAL NETWORKS

    Args:
        
    """

    def __init__(self, model, reward_type, ce_criterion, delta, optimizer, print_every, cuda=True):
        self.cuda = cuda
        self.model = model
        self.train_criterion = RLCriterion(reward_type, ce_criterion)
        self.ce_criterion = ce_criterion
        
        self.delta = delta
        self.deltas = 0

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
        
    def sample_on_batch(self, enc_data, enc_lengths, dec_data):
        # data initialization
        if self.cuda:
            enc_data, dec_data = enc_data.cuda(), dec_data.cuda()
            enc_lengths = enc_lengths.cuda()
        dec_inputs = dec_data[:, :-1]
        #target = dec_data[:, 1:].contiguous().view(-1)
        target = dec_data[:, 1:].contiguous()

        # sample data
        sample_length = min(target.size(1), self.max_sample_length)
        samples, lengths = self.sample_func(self.model, enc_data, enc_lengths, 
                self.sample_size, sample_length)

        # encoder calculation
        enc_outputs, enc_hiddens = self.model.encoder(enc_data, enc_lengths)
        #state = self.model.decoder.init_states(enc_outputs, enc_hiddens)
        
        # prepare mini sample size
        batch_size = enc_data.size(0) * self.sample_size
        mini_batch_size = enc_data.size(0) * self.mini_sample_size
        target = target.repeat(self.mini_sample_size, 1)
        ctx = enc_outputs.repeat(self.mini_sample_size, 1, 1)
        ctx_lengths = enc_lengths.repeat(self.mini_sample_size)
        #state.repeat_beam_size_times(self.mini_sample_size)

        # mini batches processing
        """
        Note: if the sample size is too big, then it is impossible to handle them all at once
        due to the limited memory of GPUs. And an alternative way is to split the batch to more
        mini-batches, we process these mini-batches step by step and accumulate the gradients, 
        then update the gradients at the end of a total batch.
        """
        loss = 0
        reward = 0
        num_mini_batches = 0
        for i in range(0, batch_size, mini_batch_size):
         
            # since state will be updated in the decoder, 
            # we need provide different states each time.
            mini_state = self.model.decoder.init_states(enc_outputs, enc_hiddens)
            mini_state.repeat_beam_size_times(self.mini_sample_size)

            mini_samples = samples[i:i+mini_batch_size]
            
            # minus 1 due to the bos
            # we keep the eos since it is better
            mini_lengths = lengths[i:i+mini_batch_size] - 1
            
            mini_inputs = mini_samples[:, :-1]
            mini_target = mini_samples[:, 1:].contiguous()
            """
            # the input of baseline network must be hidden states of RNN 
            # if we use the hidden state before linear-softmax layer, the result is bad.

            mini_outputs, mini_state = self.model.decoder(mini_inputs, ctx,
                   mini_state, ctx_lengths)
            mini_outputs = mini_outputs.view(
                    mini_target.size(0), mini_target.size(1), mini_outputs.size(-1)
            )
            mini_log_probs = self.model.generator(mini_outputs)
            baselines = self.model.baseline(mini_outputs.detach()).squeeze(-1)
            """
            state = mini_state
            outputs = []
            hiddens = []
            for i in range(mini_inputs.size(1)):
                y_t = mini_inputs[:, i:i+1]
                output_t, state = self.model.decoder(y_t, ctx, state, ctx_lengths=ctx_lengths)
                outputs.append(output_t)
                hiddens.append(state.hidden[0].detach()[-1, :, :].squeeze(0))
            mini_outputs = torch.stack(outputs, dim=1)
            mini_hiddens = torch.stack(hiddens, dim=1)
            mini_log_probs = self.model.generator(mini_outputs)
            baselines = self.model.baseline(mini_hiddens.detach()).squeeze(-1)

            # post processing
            if self.reward_type == 'ce':
                mini_loss, mini_reward = self.train_criterion(
                    mini_log_probs, target, mini_target, mini_lengths
                )
            else:
                mini_loss, mini_reward = self.train_criterion(
                    mini_log_probs, target, mini_target, mini_lengths, baselines
                )
            loss += mini_loss
            reward += mini_reward
            num_mini_batches += 1

        loss /= num_mini_batches
        reward /= num_mini_batches
        return loss, reward

    def mixer_on_batch(self, enc_data, enc_lengths, dec_data, deltas):
        # data initialization
        if self.cuda:
            enc_data, dec_data = enc_data.cuda(), dec_data.cuda()
            enc_lengths = enc_lengths.cuda()
        dec_inputs = dec_data[:, :-1]
            
        batch_size = enc_data.size(0)
        #target = dec_data[:, 1:].contiguous().view(-1)
        target = dec_data[:, 1:].contiguous()

        # encoder calculation
        enc_outputs, enc_hiddens = self.model.encoder(enc_data, enc_lengths)
        state = self.model.decoder.init_states(enc_outputs, enc_hiddens)

        # decoder ce calculation
        dec_ce_inputs = dec_inputs[:, :-deltas]
        dec_ce_target = target[:, :-deltas].contiguous()    
        dec_outputs, state = self.model.decoder(dec_ce_inputs, enc_outputs, state, ctx_lengths=enc_lengths)
        dec_ce_probs = self.model.generator(dec_outputs)

        loss_ce = self.ce_criterion(dec_ce_probs, dec_ce_target.view(-1,))
        loss_sum_ce = torch.sum(loss_ce)

        # decoder rl calculation

        y_0 = dec_inputs[:, -deltas:-(deltas-1)].repeat(self.sample_size, 1)
        ctx = enc_outputs.repeat(self.sample_size, 1, 1)
        ctx_lengths = enc_lengths.repeat(self.sample_size)
        state.repeat_beam_size_times(self.sample_size)
        target = target.repeat(self.sample_size, 1)
        batch_size = y_0.size(0)

        if self.cuda:
            y_0 = y_0.cuda()
        y_t = y_0

        # samples
        samples = []
        log_probs = []
        baselines = []

        sample_length = deltas

        not_after_eos = target[:, -deltas].ne(Constants.PAD)
        lengths = torch.ones(batch_size).type_as(y_t) * not_after_eos.long()
        before_eos = not_after_eos

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
            b_t = self.model.baseline(b_t_h)
            baselines.append(b_t)
            
            if torch.eq(y_t, Constants.EOS).type(torch.LongTensor).sum().item() == batch_size:
                break

        log_probs = torch.stack(log_probs, dim=1)
        samples = torch.cat(samples, dim=1)
        baseline = torch.cat(baselines, dim=1)
        # post processing
        dec_ce_target = target[:, :-deltas].contiguous()    
        concat_lengths = lengths + dec_ce_target.size(1)
        concat_samples = torch.cat([dec_ce_target, samples], dim=1)
        
        # sequence level
        rewards = self.train_criterion.get_bleu(target, concat_samples, concat_lengths)
        rewards = rewards[:, 0] * not_after_eos.float()
        rewards = rewards.view(-1, 1).repeat(1, samples.size(1))

        units = self.train_criterion.get_sample_unit(log_probs, samples)
        
        mask = sequence_mask(lengths, samples.size(1))

        reward_avg = (rewards * mask.float()).sum().item() / mask.float().sum().item()

        loss_b = (rewards - baseline).pow(2) * mask.float()
        loss_sum_b = torch.sum(loss_b) / batch_size
        
        rewards = rewards - baseline.detach() 

        loss_rl = - rewards * units * mask.float()
        loss_sum_rl = torch.sum(loss_rl) / batch_size

        loss = loss_sum_ce + loss_sum_b + loss_sum_rl

        return loss, reward_avg

    def train_on_epoch(self, data_iter, epoch):
        
        self.deltas += self.delta

        self.logger.info("Epoch {:02} begins training .......................".format(epoch))
        self.model.train()
        total_reward = 0.
        stats = Statistics()

        for (i, (enc_data, enc_lengths, dec_data, _)) in enumerate(data_iter):


            if self.deltas < dec_data.size(1):
                loss, reward = self.mixer_on_batch(enc_data, enc_lengths, dec_data, self.deltas)
            else:
                loss, reward = self.sample_on_batch(enc_data, enc_lengths, dec_data)
            
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
            loss = self.ce_criterion(probs, target)
            
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
            sample_length=20, mini_sample_size=10, resume_chkpt=None):
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

