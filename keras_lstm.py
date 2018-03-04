#!/usr/bin/env python3


import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Reshape, LSTM, Dense, Embedding, Dropout, Activation, Flatten
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

def get_rnn_model(max_features, max_len, in_shape):
    model = Sequential()
    #model.add(Embedding(input_dim = 1,
    #                    output_dim = 256,
    #                    input_length = max_len))
    
    model.add(LSTM(max_features, input_shape = in_shape))
    model.add(Reshape((max_len, max_features)))
    #model.add(Dropout(0.5))
    model.add(Dense(max_len, activation = 'relu'))
    model.add(Dense(max_len, activation = 'softmax'))
    #model.add(Dense(max_features))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def load_vocab(vocab_fn):
    """from lab5"""
    # idx2word is list of tokens
    with open(vocab_fn) as f:
        idx2word = [L.strip() for L in f]
    # and the inverse mapping...
    word2idx = {}
    for i, word in enumerate(idx2word):
        word2idx[word] = i
    return idx2word, word2idx

def get_lines(fn, strip = True):
    lines = []
    with open(fn) as f:
        for line in f:
            L = line.strip() if strip else line
            lines.append(L)
    return lines

def load_set(fn, max_len, word2idx):
    seq_lengths = []
    sequences = []
    with open(fn) as f:
        for line in f:
            # prepend <s>'s integer, then map remaining tokens to
            # integers and append
            token_line = [word2idx["<s>"]]
            token_line += [word2idx[word] for word in line.strip().split(' ')]
            orig_len = len(token_line)
            # </s> not in count yet
            if orig_len > max_len:
                # truncate (won't have </s>)
                new_len = max_len
                new_line = token_line[0:max_len]
            else:
                # pad with </s>
                new_len = orig_len+1
                new_line = token_line+([word2idx["</s>"]] * (max_len-orig_len))
            seq_lengths.append(new_len)
            sequences.append(new_line)
    return np.array(sequences), np.array(seq_lengths)


def make_one_hot(value, vocab_size):
    if value >= vocab_size:
        return None
    z = np.zeros(vocab_size)
    z[value] = 1
    return z

def encode_line(line, word2idx):
    V = len(word2idx)
    tokens = line.strip().split(' ')
    #enc = [float(word2idx[t])/float(V) for t in tokens]
    enc = [word2idx[t] for t in tokens]
    
    return enc

def decode_line(enc_line, idx2word):
    V = len(idx2word)
    #indices = [int(V*f) for f in enc_line]
    decoded = [idx2word[idx] for idx in enc_line]
    return " ".join(decoded)

def seq_to_one_hot_seq(vals, vocab_size):
    return [make_one_hot(v, vocab_size) for v in vals]

def main():
    train_path = "data/ptb.train.txt"
    dev_path = "data/ptb.valid.txt"
    test_path = "data/ptb.test.txt"
    vocab = "word_vocab.txt"
    max_len = 50
    idx2word, word2idx = load_vocab(vocab)
    print("loading training data...")
    #train_seqs, train_seq_lens = load_set(train_path, max_len, word2idx)
    #dev_seqs, dev_seq_lens = load_set(dev_path, max_len, word2idx)    
    est_vocab_size = 11000

    #train_seqs = train_seqs.transpose()
    #dev_seqs = dev_seqs.transpose()
    #N = train_seq_lens.shape[0]
    #V = len(idx2word) # vocab size
    print("vocab size: %d"%est_vocab_size)
    
    # load the training data the keras way
    all_lines = get_lines(train_path)
    print("line count: %s"%len(all_lines))
    if len(all_lines) > 0:
        print("first line: %s"%all_lines[0])
    else:
        print("no training data...")
        return
    #encoded_lines = [one_hot(line, est_vocab_size) for line in all_lines]
    encoded_lines = np.asanyarray([encode_line(line, word2idx) for line in all_lines])
    print("first encoded line: %s"%encoded_lines[0])
    print("... and decoded: %s"%decode_line(encoded_lines[0], idx2word))
    print("padding")

    padded_lines = pad_sequences(encoded_lines, max_len+1, padding='post')


    print("encoded count: %s"%len(encoded_lines))
    print("padded count: %s"%len(padded_lines))
    
    print("encoded_lines.shape: %s"%str(encoded_lines.shape))
    print("padded_lines.shape: %s"%str(padded_lines.shape))
    
    train_x = padded_lines[:,:len(padded_lines[0])-1]
    train_y = padded_lines[:,1:]

    train_x = train_x.reshape((-1,max_len,1))
    train_y = train_y.reshape((-1,max_len,1))

    in_shape = train_x.shape[1:]
    
    print("first x: %s"%str(tuple("%.2f"%f for f in train_x[0])))
    print("first y: %s"%str(tuple("%.2f"%f for f in train_y[0])))
    

    #train_x = train_seqs[:,:-1]
    #train_y = train_seqs[:,1:]
    print("taining model...")
    print("train_x.shape: %s"%str(train_x.shape))
    print("train_y.shape: %s"%str(train_y.shape))

    model = get_rnn_model(est_vocab_size, max_len, in_shape)
    
    model.fit(train_x, train_y, batch_size=1,
              epochs=10)
    #dev_x = train_seqs[:-1,:]
    #dev_y = train_seqs[1:,:]
    #print("evaluating...")
    #score = model.evaluate(dev_x, dev_y, batch_size=16) 
    #print(score)

if __name__ == "__main__":
    main()
