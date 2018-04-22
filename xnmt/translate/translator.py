# -*- coding: utf-8 -*-

import copy
import codecs

import torch
import torch.nn as nn
from torch.autograd import Variable

from xnmt.io.dataloader import DataLoader
from xnmt.io import Constants
from xnmt.utils import aeq, load_chkpt
from xnmt.translate.beam import Beam
from text.preprocess import convert_file_to_ids

class Translator(object):
    r"""
    Basic translator using beam-search for sequence-to-sequence model

    Args:
        model (nn.Module): the predifined model
        beam_size (int): size of beam
        n_best (int): number of translations produced
        max_length (int): maximum length output to produce
    """

    def __init__(self, model, beam_size=10, n_best=1, max_length=80):
        super(Translator, self).__init__()

        self.model = model
        self.beam_size = beam_size
        self.n_best = n_best
        self.max_length = max_length
        self.cuda = False

    def prepare_model(self, chkpt, cuda):
        _, self.model = load_chkpt(chkpt, self.model)
        if cuda:
            self.cuda = True
            self.model = self.model.cuda()
        self.model.eval()

    def prepare_vocab(self, vocab_path):
        vocab = torch.load(vocab_path)
        self.src_word2idx = vocab['src']
        self.tgt_word2idx = vocab['tgt']
        self.src_idx2word = {v:k for k,v in self.src_word2idx.items()}
        self.tgt_idx2word = {v:k for k,v in self.tgt_word2idx.items()}

    def prepare_data(self, src_path, batch_size):
        test_src = convert_file_to_ids(src_path, self.src_word2idx)
        test_tgt = copy.deepcopy(test_src)
        test_data = list(zip(test_src, test_tgt))
        test_iter = DataLoader(test_data, batch_size, False)
        self.data_iter = test_iter
        self.batch_size = batch_size

    def repeat(self, v):
        return v.repeat(1, self.beam_size, 1)

    def bottle(self, v):
        beam_size = self.beam_size
        batch_size = self.batch_size

        def be(v):
            return v.view(batch_size * beam_size, -1)

        if isinstance(v, tuple):
            return tuple(map(be, v))
        else:
            return be(v)

    def unbottle(self, v):
        beam_size = self.beam_size
        batch_size = self.batch_size

        def ube(v):
            return v.view(batch_size, beam_size, -1)

        if isinstance(v, tuple):
            return tuple(map(ube, v))
        else:
            return ube(v)

    def repeat_hiddens(self, hiddens):
        batch_size = self.batch_size
        beam_size = self.beam_size

        def r_h(h):
            # Tmp assertions: limited supported situations
            assert len(h.size()) == 2
            return h.unsqueeze(1).repeat(1, beam_size, 1).view(batch_size * beam_size, -1) 

        if isinstance(hiddens, tuple):
            return tuple(map(r_h, hiddens))
        else:
            return r_h(hiddens)

    def update_hiddens(self, hiddens, idx, pos):

        def u_h(h):
            t = h[idx, :, :]
            t.data.copy_(t.data.index_select(0, pos))
            return h

        if isinstance(hiddens, tuple):
            return tuple(map(u_h, hiddens))
        else:
            return u_h(hiddens)

    def translate_batch(self, enc_data, enc_lengths):
        """
        Args:
            enc_data (FloatTensor): (batch_size * enc_len) 
            enc_lengths (LongTensor): (batch_size)
        """
        enc_data = Variable(enc_data, volatile=True)
        if self.cuda:
            enc_data = enc_data.cuda()
            enc_lengths = enc_lengths.cuda()
        
        # Encoder
        # enc_outputs: batch_size, seq_len, hidden_dim
        enc_outputs, enc_hiddens = self.model.encoder(enc_data, enc_lengths)
        hiddens = self.model.decoder.init_states(enc_hiddens)  # (num_layers, batch_size, hidden_dim)

        # beam prepare
        hiddens = self.repeat_hiddens(hiddens)
        ctx = enc_outputs.repeat(self.beam_size, 1, 1)
        ctx_lengths = enc_lengths.repeat(self.beam_size)

        # beam search
        beams = [Beam(self.beam_size, self.cuda) for _ in range(self.batch_size)]
        for i in range(self.max_length):
            
            # decoder input data
            dec_data = torch.stack([b.get_current_state() for b in beams])  # batch_size * beam_size
            dec_data = Variable(dec_data.view(-1, 1), volatile=True)  # (batch_size * beam_size, 1)
            # decoder operation
            outputs, hiddens = self.model.decoder(dec_data, ctx, 
                    h0=hiddens, ctx_lengths=ctx_lengths)
            hiddens = self.unbottle(hiddens)

            # generator
            probs = self.model.generator(outputs) #(batch_size * beam_size, num_words)
            probs = probs.view(self.batch_size, self.beam_size, -1)

            active  = []
            for b in range(self.batch_size):
                if beams[b].done:
                    continue
                is_done = beams[b].advance(probs.data[b])
                if not is_done:
                    active += [b]
                hiddens = self.update_hiddens(hiddens, b, beams[b].get_current_origin())
            if not active:
                break
            hiddens = self.bottle(hiddens)
        
        # Get n_best results 
        all_hyps = []
        for b in range(self.batch_size):
            scores, indices = beams[b].sort_scores()
            scores = scores[:self.n_best]
            sents = [beams[b].get_hypothesis(k) for k in indices[:self.n_best]]
            #hyps = [(scores[i], sents[i]) for i in range(n_best)]
            hyps = [sents[i] for i in range(self.n_best)]
            all_hyps.append(hyps)
        return all_hyps

    def convert_ids_to_text(self, ids):
        if ids[-1] == Constants.EOS:
            ids = ids[:-1]
        return ' '.join([self.tgt_idx2word[idx] for idx in ids])

    def translate(self, opt):

        # prepare
        self.prepare_model(opt.chkpt, opt.cuda)
        self.prepare_vocab(opt.vocab)
        self.prepare_data(opt.test_src, opt.batch_size)

        # beam search
        f = codecs.open(opt.out_file, 'w', 'utf-8')
        for (enc_data, enc_lengths, _, indices) in self.data_iter:
            
            hyps = self.translate_batch(enc_data, enc_lengths)
            
            hyps = list(zip(indices, hyps))
            hyps.sort(key=lambda x: x[0])
            
            for (i, sents) in hyps:
                for sent in sents:
                    tgt = self.convert_ids_to_text(sent)
                    f.write('{}\n'.format(tgt))
        f.close()

