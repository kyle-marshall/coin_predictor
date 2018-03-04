#! /usr/bin/env python

import os
import csv

class StockData:
    stockdata = []
    def __init__(self):
        stockdata = []

    def __init__(self, path, ticker):
        self.getTrades(path, ticker) 

    def getTrades(self, path, ticker):
        if path == '':
            path = '../ExtractedData/Stocks/'
        with open(path + ticker, 'r') as stockfile:
            stockreader = csv.reader(stockfile, delimiter=',')
            for row in stockreader:
                self.stockdata.append(row) 

    def getAllStockData(self):
        return self.stockdata
    
    def getDate(self, date):
        for index, dateData in enumerate(self.stockdata):
            if dateData[0] == date:
                return dateData, index
        return None

    def checkFlux(self, date):
        index = self.getDate(date)
        i = index[1]
        
        curr = self.stockdata[i]
        prev = self.stockdata[i-1]

        if curr[1] > prev[4]:
            print("Increase")
        else:
            print("Decrease")
        
