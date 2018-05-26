# -*- coding: utf-8 -*-
import os, sys
import logging
import math, random

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, sentence_bleu

import torch
import torch.nn as nn

from xnmt.utils import drop_chkpt, load_chkpt, make_logger, Statistics, sequence_mask
from xnmt.io import Constants


class RLCriterion(object):
    """Define a reinforce criterion for computing rewards and grads in the final sampled units.
    
    Reinforce criterion is just like common criterion, like NLLLoss. And it is very easy to 
    implement with the chain rules in mind. We only need to compute the grads according to 
    famous formula in the reference paper. All the remaining operations are left to chain rules.

    And in pytorch, we just need to give gradient to the one unit in softmax which is sampled. 

    Reference paper:
        * Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning
    """
    def __init__(self, reward_type, ce_criterion):
        self.reward_type = reward_type
        if reward_type == 'bleu':
            self.get_reward = self.get_bleu
        elif reward_type == 'ce':
            self.get_reward = self.get_ce_reward
        elif reward_type == 'acc':
            self.get_reward = self.get_acc
        else:
            raise Exception('Unsupported reward type')
        # just a single scalar to keep the average of the history rewards 
        # if baseline is not given 
        self.ce_criterion = ce_criterion

    def get_bleu(self, target, sample, length):
        """Compute reward given the reference target and sampled sentence.

        Args:
            target (tensor): digits for now, since there is no difference in computing BLEU score using nltk.
                * batch_size * target_len
            sample (tensor): just like above, but is the sampled sentence.
                * batch_size * sample_len
            length (torch.LongTensor): batch_size, the effective length of each sample in a batch
        Returns:
            rewards (tensor): tensor containing reward for each sampled unit
                * batch_size * sample_len
        """
        
        batch_size, sample_len = sample.size()
        target_list = target.cpu().tolist()
        sample_list = sample.cpu().tolist()
        sample_length = length.cpu().tolist()
        target_length = [lis.index(Constants.EOS)+1 for lis in target_list]
        rewards = []
        for i in range(batch_size):
            tgt = target_list[i][:target_length[i]]
            spe = sample_list[i][:sample_length[i]]
            score = sentence_bleu([tgt], spe,
                weights = (0.25, 0.25, 0.25, 0.25),
                smoothing_function = SmoothingFunction().method2
            )
            rewards.append(score)
        rewards = torch.tensor(rewards, device=sample.device, requires_grad=False)
        rewards = rewards.view(-1, 1).repeat(1, sample_len) 
        return rewards

    def get_ce_reward(self, target, sample, length):
        """Compute reward given the reference target and sampled sentence. 

        ======================================================
        Note: This may not be compatible with the REINFORCE theory.
        ======================================================

        Performs just like cross entropy which gives word-level reward.

        Args:
            target (tensor): digits for now, since there is no difference in computing BLEU score using nltk.
                * batch_size * target_len
            sample (tensor): just like above, but is the sampled sentence.
                * batch_size * sample_len
            length (torch.LongTensor): batch_size, the effective length of each sample in a batch
        Returns:
            rewards (tensor): tensor containing reward for each sampled unit
                * batch_size * sample_len
        """
        
        batch_size, target_len = target.size()
        batch_size, sample_len = sample.size()
        if target_len >= sample_len:
            target = target[:, :sample_len]
        else:
            tmp = torch.ones((batch_size, sample_len-target_len)).type_as(sample) * -1
            target = torch.cat([target, tmp], dim=1)
        rewards = torch.eq(sample, target).type(torch.float).to(sample.device)
        mask = sequence_mask(length, sample.size(1))
        rewards = rewards * mask.float()
        """
        new_rewards = torch.ones((rewards.size(0), rewards.size(1))).type_as(rewards)
        for i in range(batch_size):
            for j in range(sample_len):
                if rewards[i, j].item() == 0.:
                    new_rewards[i, j:] = 0.
                    break
        rewards = new_rewards
        """
        return rewards

    def get_acc(self, target, sample, length):
        """Compute reward given the reference target and sampled sentence. 

        Performs just like cross entropy which gives word-level reward.

        Args:
            target (tensor): digits for now, since there is no difference in computing BLEU score using nltk.
                * batch_size * target_len
            sample (tensor): just like above, but is the sampled sentence.
                * batch_size * sample_len
            length (torch.LongTensor): batch_size, the effective length of each sample in a batch
        Returns:
            rewards (tensor): tensor containing reward for each sampled unit
                * batch_size * sample_len
        """
        
        batch_size, target_len = target.size()
        batch_size, sample_len = sample.size()
        if target_len >= sample_len:
            target = target[:, :sample_len]
        else:
            tmp = torch.ones((batch_size, sample_len-target_len)).type_as(sample) * -1
            target = torch.cat([target, tmp], dim=1)
        rewards = torch.eq(sample, target).type(torch.float).to(sample.device)
        mask = sequence_mask(length, sample.size(1))
        rewards = rewards * mask.float()
        rewards = rewards.sum(dim=1, keepdim=True) / length.view(-1, 1).float()
        rewards = rewards.repeat(1, sample_len)
        return rewards

    def get_sample_unit(self, probs, sample):
        """Get sampled unit from probs which need to backprop according to the sample

        Note: this process is the same as cross entropy, and can be implemented by NLLLoss.

        Args:
            probs (tensor): probs that output samples
                * batch_size * sample_len * vocab_size
            sample (tensor): sampled words
                * batch_size * sample_len
        Returns:
            units (tensor): tensor containing the sampled unit which need to backprop,
                similar to the loss units where gradients begin to backprop.
                * batch_size * sample_len
        """

        """
        batch_size = probs.size(0)
        sample_len = probs.size(1)
        indexes = [[], [], []]
        for i in range(batch_size):
            for j in range(sample_len):
                indexes[0].append(i)
                indexes[1].append(j)
                indexes[2].append(sample[i][j].item())
        units = probs[indexes].view(batch_size, sample_len)
        """
        batch_size = probs.size(0)
        units = self.ce_criterion(probs.view(-1, probs.size(-1)), sample.view(-1))
        units = - units.view(batch_size, -1)
        return units


    def __call__(self, probs, target, sample, length, baseline=None):
        """
        Args:
            probs (torch.Tensor): batch_size * uncertain_len * vocab_size
            target (torch.LongTensor): batch_size * seq_len
            sample (torch.LongTensor): batch_size * uncertain_len
            length (torch.LongTensor): batch_size, the effective length of each sample in a batch
            baseline (torch.Tensor): batch_size * uncertain_len

        Consideration about mask:
            * Problem: For sequence generation task, the sample of each example should be of 
            variable-length, but all examples in a batch must be the same length. So
            some samples may continue to sample even if they have output <eos> before.
            * Solution: mask all the samples after the first <eos>.
        """

        # target = target[:, :-1]  # need not to eliminate EOS
        batch_size = probs.size()[0]

        units = self.get_sample_unit(probs, sample)

        rewards = self.get_reward(target, sample, length)

        mask = sequence_mask(length, sample.size(1))


        reward_avg = rewards.sum().item()  / rewards.numel()
        if baseline is not None:
            loss_b = (rewards - baseline).pow(2) * mask.float()
            loss_sum_b = torch.sum(loss_b) / batch_size
            rewards = rewards - baseline.detach()

        #non_mask = 1 -  torch.eq(sample.data, Constants.PAD)
        #non_mask = non_mask.float()
        
        #loss = -non_mask * (rewards - self.baseline) * loss_units
        #loss = - rewards * units
        loss = - rewards * units * mask.float()
        loss_sum = torch.sum(loss) / batch_size
        if baseline is not None:
            loss_sum += loss_sum_b 
        # update baseline
        # self.baseline = self.baseline * self.ratio + new_reward_avg * (1 - self.ratio)
        return loss_sum, reward_avg


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
            b_t = self.model.baseline(b_t_h)
            baselines.append(b_t)
            
            if torch.eq(y_t, Constants.EOS).type(torch.LongTensor).sum().item() == batch_size:
                break

        log_probs = torch.stack(log_probs, dim=1)
        samples = torch.cat(samples, dim=1)
        baseline = torch.cat(baselines, dim=1)
        # post processing
        #loss, reward = self.train_criterion(log_probs, target, samples, lengths, baseline)
        loss, reward = self.train_criterion(log_probs, target, samples, lengths)
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
                    load_chkpt(resume_chkpt, self.model, self.optimizer, self.cuda)
            self.start_epoch += 1

        self.sample_size = sample_size
        self.max_sample_length = sample_length

        for epoch in range(self.start_epoch, epochs+1):
            _ = self.train_on_epoch(train_data, epoch)
            acc, ppl = self.eval_on_epoch(valid_data, epoch)
            self.epoch_step(ppl, epoch)
            drop_chkpt(epoch, self.model, self.optimizer, acc, ppl)

