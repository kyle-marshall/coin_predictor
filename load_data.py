import csv
import datetime

def getnewsdata():
	newsdata = {}
	with open('raw_data/stocknews/numnews.txt','r') as file:
		csvreader = csv.reader(file, delimiter=',')
		for row in csvreader:
			date = row[0]
			wnarray = row[1].split(' ')
			newsdata[date] = wnarray
	print(newsdata)
	return newsdata

def getstockdata(ticker):
	stockdata = {}
	with open('raw_data/Stocks/'+ticker+'.us.txt', 'r') as file:
		csvreader = csv.reader(file, delimiter=',')
		for row in csvreader:
			date = row[0]
			close = row[4]
			stockdata[date] = close
	return stockdata

def getcryptodata(coinname):
	cryptodata = {}
	with open('raw_data/cryptocurrency-financial-data/consolidated_coin_data.csv', 'r') as file:
		csvreader = csv.reader(file, delimiter=',')
		next(csvreader)
		for row in csvreader:
			coin = row[0]
			date = row[1]
			close = row[5]

			dtobj = datetime.datetime.strptime(date, '%b %d, %Y')
			date = datetime.datetime.strftime(dtobj, '%Y-%m-%d')

			cryptodata[date] = close
	return cryptodata