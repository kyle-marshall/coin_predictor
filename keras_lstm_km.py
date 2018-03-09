#!/usr/bin/env python3

from trades import StockData
from CryptoData import CryptoData
from NewsData import NewsData
import datetime
import sys
from random import randint
import numpy as np
from sklearn.metrics import confusion_matrix
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
    tech_stocks = ["fb", "googl", "intc", "amd", "nvda", "msft", "amzn", "ibm", "asx"]
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
    numWords = 5000
    while currDate <= edate:
        numDays = numDays + 1
        date = datetime.datetime.strftime(currDate, '%Y-%m-%d')

        data = []
        for ticker in tickerObjs:
            tickerData = tickerObjs[ticker].stockdata
            if date in tickerData:
                t = tickerData[date]
                data.append(t)
            else:
                data.append(-1)

        for coinName in coinObjs:
            coinData = coinObjs[coinName].cryptodata
            if date in coinData:
                c = coinData[date]
                data.append(c)
            else:
                data.append(-1)

        if date in newsdict:
            data.extend(newsdict[date])
        else:
            data.extend([0.]*numWords)
        

        currDate += datetime.timedelta(days=1)
        if currDate == edate:
            vecSize = len(data)
            print("vector size: %d" % vecSize)
            #print(data)
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
            todayData = list(allData[td])
            datLen = len(todayData)
            if not datLen == numWords + len(tech_stocks) + len(coin_names):
                print("YO!!! datLen: %d"%datLen)
                continue

            #master_x.append(list(list([d] for d in todayData)))
            master_x.append(todayData)

            threshold = .05

            todayPrice = coinObjs["bitcoin"].cryptodata[td]
            tmrwPrice = coinObjs["bitcoin"].cryptodata[tmrw]

            k = 3 # class count
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
            out = [0]*k
            out[c] = 1

            master_y.append(out)
            daysAdded = daysAdded + 1

        today += oneday
        tomorrow = today + oneday

    print("Tried %d dates, stored %d" % (daysCompared, daysAdded) )

    return master_x, master_y, vecSize, 3

def getRandomSequence(master_x, master_y, seq_length):
    start = randint(0, len(master_x) - seq_length*2)
    return master_x[start:start + seq_length], master_y[start:start + seq_length]

def prepDataForAnalysis(output):
    results = []
    for seq in range(0, len(output)):
        results.append(list(np.argmax(vec) for vec in output[seq])) 
    return np.array(results)

def analyze(pred, act):
    tot = pred.shape[0] * pred.shape[1]
    totCorrect = 0
    for seq in range(0, len(pred)):
        cm = confusion_matrix(pred[seq], act[seq], labels = [0, 1, 2])
        diag = np.diagonal(cm).copy()
        correct = np.sum(diag)
        totCorrect += correct
        print(cm)
        print(tot, totCorrect)
    return totCorrect / tot 

def main():
    master_x, master_y, vecSize, outSize = processData()
    testRat = 0.2

    # split master into train / test sets
    forTest = int(len(master_x)*testRat)
    trainX = master_x[:forTest]
    trainY = master_y[:forTest]
    testX = master_x[forTest:]
    testY = master_y[forTest:]

    """train_path = "data/ptb.train.txt"
    dev_path = "data/ptb.valid.txt"
    test_path = "data/ptb.test.txt"
    vocab = "word_vocab.txt"
    """

    max_len = 50
    trains_x = []
    trains_y = []
    sequenceLength = 50
    sequenceCount = 600
    for i in range(sequenceCount):
        x, y = getRandomSequence(trainX, trainY, sequenceLength)
        trains_x.append(x)
        trains_y.append(y)

        """
        print("--- X ---")
        print(x)
        print("--- Y ---")
        print(y)
        print("...")
        t = input()
        """
    trains_x = np.asanyarray(trains_x)
    trains_y = np.asanyarray(trains_y)

    testSeqCount = 20
    tests_x = []
    tests_y = []
    for i in range(testSeqCount):
        x, y = getRandomSequence(testX, testY, sequenceLength)
        tests_x.append(x)
        tests_y.append(y)
    tests_x = np.asanyarray(tests_x)
    tests_y = np.asanyarray(tests_y)

    print("TARGET!! %s" % str(trains_y.shape))
    
    #trains_x = trains_x.reshape((-1, sequenceLength, vecSize))
    #trains_x = trains_x.reshape((sequenceCount, sequenceLength, vecSize, 1))
    trains_y = trains_y.reshape((-1, sequenceLength, 3))

    #print("first x: %s" % str(tuple("%.2d" % f for f in trains_x[0])))
    #print("first y: %s" % str(tuple("%.2d" % f for f in trains_y[0])))
    
    in_shape = trains_x.shape[1:]
    #in_shape = trains_x.shape
    
    print("taining model...")
    print("trains_x.shape: %s" % str(trains_x.shape))
    print("trains_y.shape: %s" % str(trains_y.shape))

    print(in_shape)

    model = get_rnn_model(vecSize, sequenceLength, in_shape, 3)
    epochs = 10
    model.fit(trains_x, trains_y,
              epochs=epochs)

    # EVALUATION
    print("----IT'S EVAL TIME----")
    # to do: predict on test data instead of train data
    
    outputs = model.predict(tests_x, verbose=1)
    
    pred_class = prepDataForAnalysis(outputs)
    act_class = prepDataForAnalysis(tests_y)
    
    result = analyze(pred_class, act_class)
    print ("PERCENTAGE = %s" % result)
    
    #print("==OUTPUT==")
    #print(outputs)

    #print("classes = {0: stayed, 1: dropped, 2: rose}")
    #print("==ACTUAL CLASS==")
    #print(act_class[0])
    #print("==PREDICTED CLASS==")
    #print(pred_class[0])


if __name__ == "__main__":
    main()
