# -*- coding: utf-8 -*- 
# Optim class wrapper for torch.optim
# Largely borrow from OpenNMT-py
# https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Optim.py

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm


class Optim(object):
    """
    Wrapper for torch.optim.

    Args:
        method (str): one of [sgd, adagrad, adadelta, adam]
        lr (float): learning rate
        max_gard_norm (float): max grad norm
    """

    def __init__(self, method, lr, max_grad_norm=0, 
            lr_decay=1, start_decay_at=None,
            beta1=0.9, beta2=0.999):

        self.lr = lr
        self.method = method.lower()
        self.max_grad_norm = max_grad_norm
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.betas = [beta1, beta2]

        self._step = 0
        self.last_ppl = None
        self.start_decay = False

    def set_parameters(self, params):
        """
        Args:
            params (nn.Module.named_parameters)
        """

        self.params = []
        for k, p in params:
            if p.requires_grad:
                self.params.append(p)

        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
            # OpenNMT-py has some tricks for Adagrad. Leave it now.
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr, betas=self.betas, eps=1e-8)
            # --------------------------------------------------------
            # if the value of eps is too small, there may be numerical instability
            # --------------------------------------------------------
        else:
            raise RuntimeError("Invalid optim method: {}".format(self.method))

    def _set_rate(self, lr):
        self.lr = lr
        for op in self.optimizer.optimizers:
            op.param_groups[0]['lr'] = self.lr

    def step(self):
        self._step += 1
        if self.max_grad_norm:
            clip_grad_norm(self.params, self.max_grad_norm)
        self.optimizer.step()

    def update_learning_rate(self, ppl, epoch):
        """
        Decay learning rate if valid performance does not improve 
        or we hit the start_decay_at limit
        """
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to {}".format(self.lr))

        self.last_ppl = ppl
