import csv
import datetime

class CryptoData:

	cryptodata = {}
	mindate = ''
	maxdate = ''
	minprice = 0
	maxprice = 0

	def __init__(self):
		cryptodata = []

	def __init__(self, coinname):
		self.setdata(coinname)

	def setCryptoData(self, coinname):
		dates = []
		prices = []
		path = 'raw_data/cryptocurrency-financial-data/'
		file = 'consolidated_coin_data.csv'
		with open(path+file, 'r') as file:
			csvreader = csv.reader(file, delimiter=',')
			next(csvreader)
			for row in csvreader:
				coin = row[0]
				if coin == coinname:
					date = row[1]
					close = float(row[5])

					dtobj = datetime.datetime.strptime(date, '%b %d, %Y')
					date = datetime.datetime.strftime(dtobj, '%Y-%m-%d')

					dates.append(date)
					prices.append(close)
					self.cryptodata[date] = close
					
		self.mindate = min(dates)
		self.maxdate = max(dates)
		self.minprice = min(prices)
		self.maxprice = max(prices)

	def getCryptoData():
		return self.cryptodata