# -*- coding: utf-8 -*-
import os, sys, re, time
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from utils import aeq, sequence_mask

class Maxout(nn.Module):
    """
    Maxout Layer

    Reference: https://github.com/pytorch/pytorch/issues/805
    """

    def __init__(self, in_dim, out_dim, pool_size):
        super().__init__()
        self.in_dim, self.out_dim, self.pool_size = in_dim, out_dim, pool_size
        self.lin = nn.Linear(in_dim, out_dim * pool_size)

    def forward(self, inputs):
        """
        Args:
            inputs: (batch_size, * , in_dim)

        Returns:
            outputs: (batch_size, *, out_dim) 
    
        """
        shape = list(inputs.size())
        shape[-1] = self.out_dim
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m

if __name__ == '__main__':
    maxout = Maxout(4, 4, 2)
    ctx = Variable(torch.randn(2, 4))
    m = maxout(ctx)
    print(m)
