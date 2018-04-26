# -*- coding: utf-8 -*-
import os, sys, re, time
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from xnmt.utils import aeq, sequence_mask

class Attention(nn.Module):
    '''
    Compute attention between the context and query maxtrix.

    * Bahdanau Attention (mlp):
        * score(H_j, q) = V^T tanh(W q + U H_j])

    * Luong Attention (dot, general):
        * dot: score(H_j, q) =  q^T H_j
        * general: score(H_j, q) = q^T W H_j

    Args:
        attn_type (str): type of attention, options [dot, general, mlp]
        ctx_dim (int): dimensionality of context
        qry_dim (int): dimensionality of query

    Note: largely borrow from OpenNMT-py, https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/GlobalAttention.py
    '''
    def __init__(self, attn_type, ctx_dim, qry_dim):
        super(Attention, self).__init__()
        self.attn_type = attn_type
        
        if self.attn_type == 'general':
            self.W = nn.Linear(qry_dim, ctx_dim, bias=False)
        elif self.attn_type == 'mlp':
            self.W = nn.Linear(qry_dim, qry_dim, bias=False)
            self.U = nn.Linear(ctx_dim, qry_dim, bias=True)
            self.V = nn.Linear(qry_dim, 1, bias=False)
        else:  # dot attention requires both maxtrices have same dimensionality
            aeq(ctx_dim, qry_dim)

        self.tanh = nn.Tanh()
        self.sm = nn.Softmax(dim=-1)
        self.init_weights()

    def init_weights(self):
        """
        This initialization is especially for Bahdanau mlp attention
        """
        if self.attn_type == 'mlp':
            for name, param in self.named_parameters():
                if 1 in param.data.size() or 'bias' in name:
                    nn.init.constant(param, 0)
                else:
                    nn.init.normal(param, 0, 0.001)
        else:
            for name, param in self.named_parameters():
                if 'bias' in name:
                    nn.init.constant(param, 0)
                else:
                    nn.init.uniform(param, -0.1, 0.1)


    def score(self, qry, ctx):
        """Compute attention scores
        Args:
            qry (tensor): batch_size * qry_len * qry_dim
            ctx (tensor): batch_size * ctx_len * ctx_dim

        Returns:
            attn (tensor): batch_size * qry_len * ctx_len
            raw attention scores (unnormalized) 
        """
        qry_batch, qry_len, qry_dim = qry.size()
        ctx_batch, ctx_len, ctx_dim = ctx.size()
        aeq(qry_batch, ctx_batch)
        dim = qry_dim
        batch = qry_batch

        if self.attn_type == 'mlp':
            wq = self.W(qry.view(-1, qry_dim))
            wq = wq.view(batch, qry_len, 1, dim)
            wq = wq.expand(batch, qry_len, ctx_len, dim)

            uc = self.U(ctx.contiguous().view(-1, ctx_dim))
            uc = uc.view(batch, 1, ctx_len, dim)
            uc = uc.expand(batch, qry_len, ctx_len, dim)

            wquc = self.tanh(wq + uc)
            
            attn = self.V(wquc.view(-1, dim)).view(batch, qry_len, ctx_len)
        elif self.attn_type == 'dot':
            aeq(ctx_dim, qry_dim)
            ctx = ctx.transpose(1, 2)
            # (batch_size, qry_len, qry_dim) * (batch_size, ctx_dim, ctx_len) --> (batch_size, qry_len, ctx_len)
            attn = torch.bmm(qry, ctx)
        else:
            # qry = qry.view(qry_batch * qry_len, qry_dim)
            # qry = self.W(qry).view(qry_batch, qry_len, ctx_dim) 
            # the linear module now supports multiple dimensions instead of only 2 dimensions
            qry = self.W(qry) 
            ctx = ctx.transpose(1, 2)
            attn = torch.bmm(qry, ctx)
        return attn

    def forward(self, qry, ctx, ctx_lengths=None):
        """Compute attention scores
        Args:
            qry (tensor): batch_size * qry_len * qry_dim
            ctx (tensor): batch_size * ctx_len * ctx_dim
            ctx_lengths (LongTensor): [batch_size] containing ctx lengths
        Returns:
            attn (tensor): batch_size * qry_len * ctx_len, normalized attention scores
            attn_ctx (tensor): batch_size * qry_len * ctx_dim, attentioned context matrix
            note: if qry_len == 1, then the dimension is squeezed
        """
        # one step input
        if qry.dim() == 2:
            one_step = True
            qry = qry.unsqueeze(1)
        else:
            one_step = False

        align = self.score(qry, ctx)
        if ctx_lengths is not None:
            mask = sequence_mask(ctx_lengths)
            mask = mask.unsqueeze(1)  # batch_size * 1 * ctx_len, make it broadcastable
            align.data.masked_fill_(1 - mask, -float('inf')) 
        attn = self.sm(align) # batch_size * qry_len * ctx_len
        attn_ctx = torch.bmm(attn, ctx) # batch_size * qry_len * ctx_dim

        if one_step:
            attn = attn.squeeze(1)
            attn_ctx = attn_ctx.squeeze(1)
        return attn, attn_ctx  

