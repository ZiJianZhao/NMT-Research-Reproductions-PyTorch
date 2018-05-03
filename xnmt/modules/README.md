# modules

Although the code are heavily borrowed from OpenNMT-py, there are some differences in the choice of design.

# Attention

The OpenNMT-py concat the query matrix and the context matrix to output by default. 
However, in our attention, we just calculate the weighted context matrix. All other operations are left for decoder.

# 2018-05-03 Update

The original decoder.py is splited into 2 files: decoder.py and old_decoder.py.

The new decoder.py contains decoder classes like OpenNMT-py while the old_decoder.py contains the
basic seq2seq class and rnnsearch. However, I can't get expected results using rnnsearch in a 
chinese-english translation task.


