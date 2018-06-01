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
from xnmt.translate.translator import Translator

def sample_on_batch(model, enc_data, enc_lengths, sample_size, sample_length):
    """Sample from a model on input data
    Args:
        * 
    Returns:
        * samples (torch.FloatTensor, batch_size * sample_size * seq_len) : generated samples
        * lengths (torch.LongTensor, (batch_size*sample_size)): effective lengths before eos
    """
    # encoder calculation
    enc_outputs, enc_hiddens = model.encoder(enc_data, enc_lengths)
    state = model.decoder.init_states(enc_outputs, enc_hiddens)

    # prepare sample size
    batch_size = enc_data.size(0) * sample_size
    ctx = enc_outputs.repeat(sample_size, 1, 1)
    ctx_lengths = enc_lengths.repeat(sample_size)
    state.repeat_beam_size_times(sample_size)

    # sample
    y_t = torch.LongTensor([Constants.BOS for _ in range(batch_size)]).view(-1, 1).type_as(enc_data)
    samples = [y_t]

    # judge effective lengths according to the first generated <eos>
    lengths = torch.ones(batch_size).type_as(y_t)
    before_eos = torch.ones(batch_size).type_as(y_t).byte()

    t = 0
    while t < sample_length:
        t += 1

        # decoder step forward
        outputs, state = model.decoder(y_t, ctx, 
                state, ctx_lengths=ctx_lengths)
        log_prob_t = model.generator(outputs)  # batch_size * vocab_size

        # sample next step inputs
        prob_t = torch.exp(log_prob_t)

        y_t = prob_t.multinomial(1, replacement=True)
        y_t = y_t.detach()  
        samples.append(y_t)  # batch_size * 1

        # eos judgement
        lengths += before_eos.long()
        before_eos = torch.ne(y_t.view(-1), Constants.EOS) & before_eos

        if torch.eq(y_t, Constants.EOS).type(torch.LongTensor).sum().item() == batch_size:
            break

    samples = torch.cat(samples, dim=1)
    mask = sequence_mask(lengths, samples.size(1))
    samples.masked_fill_(1-mask, 0)

    return samples, lengths

def beam_search_on_batch(model, enc_data, enc_lengths, sample_size, sample_length):
    """Beam search from a model on input data
    Previous wrong consideration:
        * Need indices to align the enc data dn generated dec data
    Correct explanation:
        * Since the enc_data is already in expected order, need not to align
    Args:
        * 
    Returns:
        * samples (torch.FloatTensor, batch_size * sample_size * seq_len) : generated samples
        * lengths (torch.LongTensor, (batch_size*sample_size)): effective lengths before eos
    """
    # beam size is at least 10
    beam_size = max(sample_size, 10)
    translator = Translator(model, beam_size, sample_size, sample_length)
    translator.cuda = enc_data.is_cuda
    translator.batch_size = enc_data.size(0)

    hyps = translator.translate_batch(enc_data, enc_lengths)
    hyps = [hyps[i][j] for j in range(len(hyps[0])) for i in range(len(hyps))]

    lengths = [len(s) for s in hyps]
    max_len = max(lengths)
    lengths = torch.tensor(lengths) + 1

    hyps = [[Constants.BOS] + s + [Constants.PAD] * (max_len - len(s)) for s in hyps]

    samples = torch.tensor(hyps).type_as(enc_data)
    lengths = lengths.type_as(enc_data)

    return samples, lengths


