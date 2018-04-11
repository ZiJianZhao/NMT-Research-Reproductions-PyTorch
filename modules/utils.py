# -*- coding: utf-8 -*-
import torch


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
            "Not all arguments have the same value: " + str(args)

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    Borrow from OpenNMT-py: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Utils.py

    Args:
        lengths (LongTensor): containing sequence lengths
        max_len (int): maximum padded length
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

if __name__ == '__main__':
    a = torch.LongTensor([1,2,3])
    print(sequence_mask(a))
    aeq(32, 31)
