#!/usr/bin/env python3

from trades import StockData
from CryptoData import CryptoData
from NewsData import NewsData
import datetime
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Reshape, LSTM, Dense, Embedding, Dropout, Activation, Flatten, Masking
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

def get_rnn_model(max_features, max_len, in_shape):
    model = Sequential()

    model.add(Masking(mask_value = -1, input_shape = in_shape))
    model.add(LSTM(max_features, input_shape = in_shape))
    model.add(Reshape((max_len, max_features)))
    #model.add(Dropout(0.5))
    model.add(Dense(max_len, activation = 'relu'))
    model.add(Dense(max_len, activation = 'softmax'))
    #model.add(Dense(max_features))
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def processData():
    allData = []
    tech_stocks = ["fb", "googl", "intc", "amd", "nvda"]
    coin_names = ["bitcoin", "ethereum"]
    path = '../ExtractedData/Stocks/'

    startDate = "0000"
    endDate = "9999"
    maxTickerPrice = 0
    maxCoinPrice = 0
    vecSize = 0

    # get stock dictionary
    tickerDicts = {}
    for ticker in tech_stocks:
        s = StockData(ticker)
        s.normalize()
        if s.mindate > startDate:
            startDate = s.mindate
        if s.maxdate < endDate:
            endDate = s.maxdate
        if s.maxprice > maxTickerPrice:
            maxTickerPrice = s.maxprice
        tickerDicts[ticker] = s

    # get coin dictionary
    coinDicts = {}
    for coin in coin_names:
        c = CryptoData(coin)
        c.normalize()
        if c.mindate > startDate:
            startDate = c.mindate
        if c.maxdate < endDate:
            endDate = c.maxdate
        if c.maxprice > maxCoinPrice:
            maxCoinPrice = c.maxprice
        coinDicts[coin] = c

    # get headline dictionary
    n = NewsData()
    newsdict = n.getNewsData()
    if n.mindate > startDate:
        startDate = n.mindate
    if n.maxdate < endDate:
        endDate = n.maxdate

    print("Date range: %s, %s" % (startDate,endDate))


    # combine values from dictionary by date
    startParts = map(int, startDate.split("-"))
    year, month, day = list(startParts)
    sdate = datetime.date(year, month, day)

    endParts = map(int, endDate.split("-"))
    year, month, day = list(endParts)
    edate = datetime.date(year, month, day)
    
    currDate = sdate
    allData = {}
    while currDate <= edate:
        date = "%d-%d-%d" % (currDate.year, currDate.month, currDate.day) 

        data = []
        for ticker in tickerDicts:
            if date in tickerDicts:
                t = ticker[date]
                data.append(t / maxTickerPrice)
            else:
                data.append(-1)

        for coin in coinDicts:
            if date in coinDicts:
                c = coinDicts[date]
                data.append(c / maxCoinPrice)
            else:
                data.append(-1)

        if date in newsdict:
            data.append(newsdict[date])
        else:
            data.append(-1)

        currDate += datetime.timedelta(days=1)
        if currDate == edate:
            vecSize = len(data)
            print("vector size: %d" % vecSize)
        allData[date] = data

    oneday = datetime.timedelta(days=1)
    today = sdate
    tomorrow = today + oneday
    master_x = []
    master_y = []
    
    while today <= edate - oneday:
        td = datetime.datetime.strftime(today, '%Y-%m-%d')
        tmrw = datetime.datetime.strftime(tomorrow, '%Y-%m-%d')
        if td in allData and tmrw in allData:
            master_x.append(allData[td])
            tmrwData = allData[tmrw]
            tmrwPrices = tmrwData[len(tech_stocks):-1]
            master_y.append(tmrwPrices)

        today += oneday
        tomorrow = today + oneday

    return master_x, master_y, vecSize

def main():
    master_x, master_y, vecSize = processData()

    """train_path = "data/ptb.train.txt"
    dev_path = "data/ptb.valid.txt"
    test_path = "data/ptb.test.txt"
    vocab = "word_vocab.txt"
    """
    # we will chop our training data into sequences of this size
    max_len = 50
    #train_seqs, train_seq_lens = load_set(train_path, max_len, word2idx)
    #dev_seqs, dev_seq_lens = load_set(dev_path, max_len, word2idx)
    #est_vocab_size = 11000

    #train_seqs = train_seqs.transpose()
    #dev_seqs = dev_seqs.transpose()
    #N = train_seq_lens.shape[0]
    #V = len(idx2word) # vocab size

    #train_size =
    #test_size =
    #train, test =

    # load the training data the keras way
    #encoded_lines = [one_hot(line, est_vocab_size) for line in all_lines]

    

    #padded_lines = pad_sequences(encoded_lines, max_len + 1, padding='post')

    #train_x = padded_lines[:,:len(padded_lines[0]mp) - 1]
    #train_y = padded_lines[:,1:]

    test_size =  int(len(master_x)*.8)
    
    train_x = master_x[:test_size]
    train_y = master_y[:test_size]

    test_x = master_x[test_size:]
    test_y = master_y[test_size:]

    #!!! PROBLEMS HERE!!!
    train_x = train_x.reshape((-1,max_len,1))
    train_y = train_y.reshape((-1,max_len,1))

    in_shape = train_x.shape[1:]
    
    print("first x: %s" % str(tuple("%.2f" % f for f in train_x[0])))
    print("first y: %s" % str(tuple("%.2f" % f for f in train_y[0])))
    
    #train_x = train_seqs[:,:-1]
    #train_y = train_seqs[:,1:]
    print("taining model...")
    print("train_x.shape: %s" % str(train_x.shape))
    print("train_y.shape: %s" % str(train_y.shape))

    model = get_rnn_model(max_len, in_shape)
    
    model.fit(train_x, train_y, batch_size=1,
              epochs=10)
    #dev_x = train_seqs[:-1,:]
    #dev_y = train_seqs[1:,:]
    #print("evaluating...")
    #score = model.evaluate(dev_x, dev_y, batch_size=16)
    #print(score)
if __name__ == "__main__":
    main()
