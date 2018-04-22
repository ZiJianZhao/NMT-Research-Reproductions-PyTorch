# -*- coding: utf-8 -*-

import argparse
import os, sys
import codecs
from collections import Counter

import torch

sys.path.append('../')
from xnmt.io import Constants
from xnmt.utils import make_logger

def preprocess_opts(parser):
    # Data options

    parser.add_argument('-train_src', required=True,
                       help="Path to the training source data")
    parser.add_argument('-train_tgt', required=True,
                       help="Path to the training target data")
    parser.add_argument('-valid_src', required=True,
                       help="Path to the validation source data")
    parser.add_argument('-valid_tgt', required=True,
                       help="Path to the validation target data")

    parser.add_argument('-save_data', required=True,
                       help="Output file for the prepared data")

    # Dictionary options, for text corpus
    parser.add_argument('-src_vocab_size', type=int, default=30000,
                       help="Size of the source vocabulary")
    parser.add_argument('-tgt_vocab_size', type=int, default=30000,
                       help="Size of the target vocabulary")
    parser.add_argument('-share_vocab', action='store_true',
                       help="Share source and target vocabulary")

    # Truncation options, for text corpus
    parser.add_argument('-src_seq_length', type=int, default=50,
                       help="Maximum source sequence length")
    parser.add_argument('-tgt_seq_length', type=int, default=50,
                       help="Maximum target sequence length to keep.")

    # Data processing options
    parser.add_argument('-seed', type=int, default=3435,
                       help="Random seed")


def parse_args():
    parser = argparse.ArgumentParser(
            description='Preprocess Options',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    preprocess_opts(parser)
    opt = parser.parse_args()

    # fix random seed
    torch.manual_seed(opt.seed)
    
    return opt

def read_file(filename, max_sent_len):
    """
    converst the tokenized sentences into word lists.

    Args:
        filename (str) : filename containing text
        max_sent_len (int) : filter sentences that are too long

    Returns:
        word_lists (list): list containing word lists
    """
    word_lists = []
    filter_sents = 0
    with codecs.open(filename, 'r', 'utf-8') as f:
        for sent in f:
            words = sent.strip().split()
            if len(words) > max_sent_len:
                filter_sents += 1
            word_lists.append(words)
    print('Total lines: {}'.format(len(word_lists)))
    print('Filter lines: {}'.format(filter_sents))
    return word_lists

def build_vocab(filename, vocab_size):
    words = []
    #with codecs.open(filename, 'r', 'utf-8') as f:
    with open(filename, 'r', encoding='utf-8') as f:
        for sent in f:
            ws = sent.strip().split()
            words.extend(ws)
    counter = Counter(words)
    vocab_freq = counter.most_common(vocab_size)
    word2idx = {
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK,
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS
    }
    for word, freq in vocab_freq:
        word2idx[word] = len(word2idx)
    print("Vocab size: {}".format(len(word2idx)))

    return word2idx


def convert_file_to_ids(filename, word2idx):

    # ********************************************************************************
    # --------------------------------------------------------------------------------
    # Note: using codecs.open causes readlines output one more line than it should be. 
    # It results that the number of lines in train.src is not compatible with that in train.tgt
    # --------------------------------------------------------------------------------
    # ********************************************************************************

    #with codecs.open(filename, 'r', 'utf-8') as f:
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        lists = [line.strip().split() for line in lines]
    return [[word2idx.get(w) if w in word2idx else Constants.UNK for w in sent] for sent in lists]

def main():

    logger = make_logger('log.preprocess')

    # parse arguments
    opt = parse_args()
    dir_name, file_name = os.path.split(opt.save_data)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    # build and save vocab
    print("Building vocab...")
    src_word2idx = build_vocab(opt.train_src, opt.src_vocab_size)
    tgt_word2idx = build_vocab(opt.train_tgt, opt.src_vocab_size)
    vocab = {'src': src_word2idx, 'tgt': tgt_word2idx}
    logger.info('Src vocab size: {}'.format(len(src_word2idx)))
    logger.info('Tgt vocab size: {}'.format(len(tgt_word2idx)))
    torch.save(vocab, opt.save_data+'.vocab.pt')

    # convert train text to ids
    print("Converting train text to ids...")
    train_src = convert_file_to_ids(opt.train_src, src_word2idx)
    train_tgt = convert_file_to_ids(opt.train_tgt, tgt_word2idx)
    print(len(train_src), len(train_tgt))
    assert len(train_src) == len(train_tgt)
    train = list(zip(train_src, train_tgt))
    logger.info("Train total lines: {}".format(len(train)))
    train = [t for t in train if len(t[0]) <= opt.src_seq_length and len(t[1]) <= opt.tgt_seq_length]
    logger.info("Train after filtering (src: {}, tgt: {}): {}".format(opt.src_seq_length, opt.tgt_seq_length, len(train)))
    torch.save(train, opt.save_data+'.train.pt')

    # convert dev text to ids
    print("Converting valid text to ids...")
    valid_src = convert_file_to_ids(opt.valid_src, src_word2idx)
    valid_tgt = convert_file_to_ids(opt.valid_tgt, tgt_word2idx)
    assert len(valid_src) == len(valid_tgt)
    valid = list(zip(valid_src, valid_tgt))
    logger.info("Valid total lines: {}".format(len(valid)))
    torch.save(valid, opt.save_data+'.valid.pt')

if __name__ == '__main__':
    main()
