# modules

Although the code are heavily borrowed from OpenNMT-py, there are some differences

# Attention

The OpenNMT-py concat the query matrix and the context matrix to output by default. 
However, in our attention, we just calculate the weighted context matrix. All other operations are left for decoder.




