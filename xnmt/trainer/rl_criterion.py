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

        rewards = self.get_reward(target, sample, length).float()
        #rewards = (self.get_acc(target, sample, length).float() + self.get_bleu(target, sample, length).float())

        mask = sequence_mask(length, sample.size(1))

        reward_avg = (rewards * mask.float()).sum().item()  / mask.float().sum().item()
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

