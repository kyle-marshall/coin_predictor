#!/usr/bin/env python3

from trades import StockData
from CryptoData import CryptoData
from NewsData import NewsData
import datetime
import sys
from random import randint
import numpy as np
from keras.models import Sequential
from keras.layers import Reshape, LSTM, Dense, Embedding, Dropout, Activation, Flatten, Masking
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

def get_rnn_model(max_features, max_len, in_shape, outUnits):
    model = Sequential()

    model.add(Masking(mask_value = -1, input_shape = in_shape))
    model.add(LSTM(10, input_shape = in_shape))
    model.add(Dense(max_len*outUnits, activation = 'relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(max_len, activation = 'relu'))
    model.add(Dense(max_len*outUnits, activation = 'softmax'))
    model.add(Reshape((max_len, outUnits)))
    #model.add(Dense(max_features))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def processData():
    allData = []
    tech_stocks = ["fb", "googl", "intc", "amd", "nvda"]
    coin_names = ["bitcoin"]
    path = '../ExtractedData/Stocks/'

    startDate = "0000"
    endDate = "9999"
    maxTickerPrice = 0
    maxCoinPrice = 0
    vecSize = 0

    # get stock dictionary
    tickerObjs = {}
    for ticker in tech_stocks:
        s = StockData(ticker)
        s.normalize()
        if s.mindate > startDate:
            startDate = s.mindate
        if s.maxdate < endDate:
            endDate = s.maxdate
        if s.maxprice > maxTickerPrice:
            maxTickerPrice = s.maxprice
        tickerObjs[ticker] = s

    # get coin dictionary
    coinObjs = {}
    for coin in coin_names:
        c = CryptoData(coin)
        c.normalize()
        if c.mindate > startDate:
            startDate = c.mindate
        if c.maxdate < endDate:
            endDate = c.maxdate
        if c.maxprice > maxCoinPrice:
            maxCoinPrice = c.maxprice
        coinObjs[coin] = c

    # get headline dictionary
    n = NewsData()
    n.normalize()
    newsdict = n.getNewsData()
    if n.mindate > startDate:
        startDate = n.mindate
    if n.maxdate < endDate:
        endDate = n.maxdate

    
    # combine values from dictionary by date
    startParts = map(int, startDate.split("-"))
    year, month, day = list(startParts)
    sdate = datetime.date(year, month, day)

    endParts = map(int, endDate.split("-"))
    year, month, day = list(endParts)
    edate = datetime.date(year, month, day)
    
    print("Date range: %s, %s" % (startDate,endDate))

    currDate = sdate
    allData = {}
    numDays = 0
    while currDate <= edate:
        numDays = numDays + 1
        date = datetime.datetime.strftime(currDate, '%Y-%m-%d')

        data = []
        for ticker in tickerObjs:
            tickerData = tickerObjs[ticker].stockdata
            if date in tickerData:
                t = tickerData[date]
                data.append(t / maxTickerPrice)
            else:
                data.append(-1)

        for coinName in coinObjs:
            coinData = coinObjs[coinName].cryptodata
            if date in coinData:
                c = coinData[date]
                data.append(c / maxCoinPrice)
            else:
                data.append(-1)

        if date in newsdict:
            data.extend(newsdict[date])
        else:
            data.append(-1)

        currDate += datetime.timedelta(days=1)
        if currDate == edate:
            vecSize = len(data)
            print("vector size: %d" % vecSize)
            print(data)
        #print("one date vector %s"%data)
        allData[date] = data

    oneday = datetime.timedelta(days=1)

    startParts = map(int, startDate.split("-"))
    year, month, day = list(startParts)
    sdate = datetime.date(year, month, day)

    today = sdate

    tomorrow = today + oneday
    master_x = []
    master_y = []
    
    daysCompared = 0
    daysAdded = 0
    while today < edate:
        daysCompared = daysCompared + 1
        td = datetime.datetime.strftime(today, '%Y-%m-%d')
        tmrw = datetime.datetime.strftime(tomorrow, '%Y-%m-%d')
        if td in allData and tmrw in allData:
            todayData = allData[td]
            master_x.append(todayData)

            threshold = .05

            todayPrice = coinObjs["bitcoin"].cryptodata[td]
            tmrwPrice = coinObjs["bitcoin"].cryptodata[tmrw]

            c = 0
            if (abs(tmrwPrice-todayPrice) < threshold*todayPrice ):
                #same
                c = 0
            elif(todayPrice > tmrwPrice):
                #dropped
                c = 1
            elif(todayPrice < tmrwPrice):
                #increase
                c = 2
            out = [0]*c + [1] + [0]*(2-c)

            master_y.append(out)
            daysAdded = daysAdded + 1

        today += oneday
        tomorrow = today + oneday

    print("Tried %d dates, stored %d" % (daysCompared, daysAdded) )

    return master_x, master_y, vecSize, 1

def getRandomSequence(master_x, master_y, seq_length):
    start = randint(0, len(master_x) - seq_length*2)
    return master_x[start:start + seq_length], master_y[start:start + seq_length]

def main():
    master_x, master_y, vecSize, outSize = processData()

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

    #test_size =  int(len(master_x)*.8)
    
    #train_x = np.array(master_x[:test_size])
    #print(train_x)
    #train_y = np.array(master_y[:test_size])

    #test_x = np.array(master_x[test_size:])
    #test_y = np.array(master_y[test_size:])

    #!!! PROBLEMS HERE!!!
    #train_x = train_x.reshape((len(train_x),1,1))
    #train_y = train_y.reshape((len(train_y),1,1))

    #in_shape = vecSize
    
    #print("first x: %s" % str(tuple("%.2f" % f for f in train_x[0])))
    #print("first y: %s" % str(tuple("%.2f" % f for f in train_y[0])))
    
    #train_x = train_seqs[:,:-1]
    #train_y = train_seqs[:,1:]
    
    trains_x = []
    trains_y = []
    sequenceLength = 20
    sequenceCount = 10
    for i in range(sequenceCount):
        x, y = getRandomSequence(master_x, master_y, sequenceLength)
        trains_x.append(x)
        trains_y.append(y)
    
    trains_x = np.asanyarray(trains_x)
    trains_y = np.asanyarray(trains_y)

    print("TARGET!! %s" % str(trains_y.shape))
    
    trains_x = trains_x.reshape((-1, sequenceLength, vecSize))
    trains_y = trains_y.reshape((-1, sequenceLength, 3))

    #print("first x: %s" % str(tuple("%.2d" % f for f in trains_x[0])))
    #print("first y: %s" % str(tuple("%.2d" % f for f in trains_y[0])))
    
    in_shape = trains_x.shape[1:]

    print("taining model...")
    print("trains_x.shape: %s" % str(trains_x.shape))
    print("trains_y.shape: %s" % str(trains_y.shape))

    print(in_shape)

    model = get_rnn_model(vecSize, sequenceLength, in_shape, 3)
    
    model.fit(trains_x, trains_y, batch_size=1,
              epochs=10)
    #dev_x = train_seqs[:-1,:]
    #dev_y = train_seqs[1:,:]
    #print("evaluating...")
    #score = model.evaluate(dev_x, dev_y, batch_size=1)
    #print(score)
    x = model.predict(np.array([trains_x[0]]))
    print(x)
    print(trains_y[0])


if __name__ == "__main__":
    main()
