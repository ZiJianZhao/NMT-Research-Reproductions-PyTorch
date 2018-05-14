# -*- coding: utf-8 -*-

import random
import numpy as np

import torch
from torch.autograd import Variable

from xnmt.io import Constants

class DataLoader(object):
    """
    Define an ordered iterator for encoder-decoder framework

    Args:
        data (list of tuples): containing tuples as (enc_ids, dec_ids), the converted text ids.
        batch_size (int): batch size.
        epoch_shuffle (bool, optional): whether shuffle the whole dataset at the begining of each epoch.
        batch_sort (bool, optional): whether sort the data insides a batch, if you want to use the variable-length rnn, set it True. 
        
        Note: Currently, the batch_sort is default (only) to True to eliminate the effects of padding.

    Returns: (generator)
        enc_data (torch.LongTensor): batch_size * max_enc_len
        enc_length (torch.LongTensor): [batch_size], containing lengths of each line in a batch
        dec_data (torch.LongTensor): batch_size * max_dec_len
        indices (list of ints): record the sorted indices inside of a batch to recover the original order.
            Especially useful when we decode the test data.
    """

    def __init__(self, data, batch_size, epoch_shuffle=True, batch_sort=True):

        super(DataLoader, self).__init__()
        self.data = data
        self.data_len = len(self.data)
        self.batch_size = batch_size
        self.epoch_shuffle = epoch_shuffle
        self.batch_sort = batch_sort
        self.indices = list(range(self.data_len))
        self.batches = (self.data_len + self.batch_size - 1) // self.batch_size 
        self.reset()

    def reset(self):
        self.idx = 0
        if self.epoch_shuffle:
            random.shuffle(self.indices)
    
    def __len__(self):
        return self.batches

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.data_len:
            self.reset()
            raise StopIteration
        index = self.indices[self.idx:min(self.idx+self.batch_size, self.data_len)]
        data = [self.data[i] for i in index]
        indices = list(range(len(index)))
        tmp = list(zip(data, indices))
        tmp.sort(key=lambda x: len(x[0][0]), reverse=True)
        enc = [tmp[i][0][0] for i in range(len(tmp))]
        dec = [tmp[i][0][1] for i in range(len(tmp))]
        indices = [tmp[i][1] for i in range(len(tmp))]
        enc_length = torch.LongTensor([len(enc[i]) for i in range(len(enc))])
        enc_len = max([len(l) for l in enc])
        dec_len = max([len(l) for l in dec])
        enc = [l + [Constants.PAD for i in range(enc_len-len(l))] for l in enc]
        dec = [ [Constants.BOS] + l + [Constants.EOS] + [Constants.PAD for i in range(dec_len-len(l))] for l in dec]
        enc = np.asarray(enc, dtype='int64')
        dec = np.asarray(dec, dtype='int64')
        enc_data = torch.from_numpy(enc)
        dec_data = torch.from_numpy(dec)
        self.idx += len(tmp)
        return enc_data, enc_length, dec_data, indices

class PreNDataLoader(object):
    """
    Define an ordered iterator with pre-fetch N sorting batches for encoder-decoder framework

    Args:
        data (list of tuples): containing tuples as (enc_ids, dec_ids), the converted text ids.
        batch_size (int): batch size.
        pre_fetch (int, optional, 20): N, the number of batches to be pre-fetched.
        epoch_shuffle (bool, optional): whether shuffle the whole dataset at the begining of each epoch.
        batch_sort (bool, optional): whether sort the data insides a batch, if you want to use the variable-length rnn, set it True. 
        
        Note: Currently, the batch_sort is default (only) to True to eliminate the effects of padding.

    Returns: (generator)
        enc_data (torch.LongTensor): batch_size * max_enc_len
        enc_length (torch.LongTensor): [batch_size], containing lengths of each line in a batch
        dec_data (torch.LongTensor): batch_size * max_dec_len
        indices (list of ints): record the sorted indices inside of a batch to recover the original order.
            Especially useful when we decode the test data.
    """

    def __init__(self, data, batch_size, pre_fetch=20, epoch_shuffle=True, batch_sort=True):

        super(PreNDataLoader, self).__init__()
        self.data = data
        self.pre_fetch = pre_fetch
        self.data_len = len(self.data)
        self.batch_size = batch_size
        self.epoch_shuffle = epoch_shuffle
        self.batch_sort = batch_sort
        self.indices = list(range(self.data_len))
        self.batches = (self.data_len + self.batch_size - 1) // self.batch_size 
        self.reset()

    def reset(self):
        self.idx = 0
        if self.epoch_shuffle:
            random.shuffle(self.indices)
    
    def __len__(self):
        return self.batches

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.data_len:
            self.reset()
            raise StopIteration
        if self.idx % (self.batch_size * self.pre_fetch) == 0:
            self.pre_idx = self.idx
            index = self.indices[self.idx:min(self.idx + self.batch_size * self.pre_fetch, self.data_len)]
            datas = [self.data[i] for i in index]
            indices = list(range(len(index)))
            self.tmp = list(zip(datas, indices))
            self.tmp.sort(key=lambda x: len(x[0][0]), reverse=True)
        beg_idx = self.idx - self.pre_idx
        end_idx = min(self.idx - self.pre_idx + self.batch_size, len(self.tmp))
        enc = [self.tmp[i][0][0] for i in range(beg_idx, end_idx)]
        dec = [self.tmp[i][0][1] for i in range(beg_idx, end_idx)]
        indices = [self.tmp[i][1] for i in range(beg_idx, end_idx)]
        enc_length = torch.LongTensor([len(enc[i]) for i in range(len(enc))])
        enc_len = max([len(l) for l in enc])
        dec_len = max([len(l) for l in dec])
        enc = [l + [Constants.PAD for i in range(enc_len-len(l))] for l in enc]
        dec = [ [Constants.BOS] + l + [Constants.EOS] + [Constants.PAD for i in range(dec_len-len(l))] for l in dec]
        enc = np.asarray(enc, dtype='int64')
        dec = np.asarray(dec, dtype='int64')
        enc_data = torch.from_numpy(enc)
        dec_data = torch.from_numpy(dec)
        self.idx += (end_idx - beg_idx)
        return enc_data, enc_length, dec_data, indices

