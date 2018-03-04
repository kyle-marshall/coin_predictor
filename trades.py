#! /usr/bin/env python

import os
import csv

class StockData:

    stockdata = {}
    mindate = ''
    maxdate = ''
    minprice = 0
    maxprice = 0

    def __init__(self):
        stockdata = []

    def __init__(self, ticker):
        self.getTrades(ticker)

    def getTrades(self, ticker):
        dates = []
        prices = []
        with open('raw_data/Stocks/'+ticker+'.us.txt', 'r') as stockfile:
            stockreader = csv.reader(stockfile, delimiter=',')
            for row in stockreader:
                date = row[0]
                close = float(row[4])

                dates.append(date)
                prices.append(close)
                self.stockdata[date] = close
                
        self.mindate = min(dates)
        self.maxdate = max(dates)
        self.minprice = min(prices)
        self.maxprice = max(prices)

    def getAllStockData(self):
        return self.stockdata

        
