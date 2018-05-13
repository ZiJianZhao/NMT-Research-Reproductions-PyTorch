# NMT-Research-Reproductions-PyTorch

This is a repository for reproducing research results in neural machine translation. All details are implemented in [PyTorch](http://pytorch.org).

# xnmt

Define general interfaces and modules for NMT. Heavily borrowed from [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py).

# text

Contains the preprocess script for texts.

# RNNsearch

Reference paper:

> [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)

Comment: I can not get expected performance in a Chinese-English translation task.

# Base-Model

Reference paper:

> [Effective Approaches to Attention-based Neural Machine Translation](http://aclweb.org/anthology/D15-1166)

Comment: Self-defined multi-layer encoder-decoder with attention and input feeding. The model can get good performance in
 a Chinese-English translation task. And it is adopted as baseline for following experiments.


# BPE

Reference paper:

> [Neural Machine Translation of Rare Words with Subword Units](http://www.aclweb.org/anthology/P16-1162)

Comment: It just has the preprocess script since all others are just the same as normal procedure.
