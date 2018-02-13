#!/usr/bin/env python3


words_to_ignore = [
    "the","in","a","an","from","to","and","then"
    ]

def load_vocab(vocab_fn):
    """
    Reads the vocab file, producing a list that maps for integer to token
    and a dictionary that maps from token to integer.

    :param vocab_fn: (string) the filename of the vocab file

    :return: (list,dictionary) the idx2word list and word2idx dictionary
    """

    # idx2word is a just a list of the vocab tokens
    with open(vocab_fn) as f:
        idx2word = [line.strip() for line in f]

    # word2idx is the inverse mapping
    word2idx = {}
    for i,word in enumerate(idx2word):
        word2idx[word] = i

    return idx2word,word2idx
