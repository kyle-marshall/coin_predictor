import csv
import datetime

class NewsData:

	newsdata = {}
	mindate = ''
	maxdate = ''

	def __init__(self):
		self.setNewsData()

	def setNewsData(self):
		dates = []
		path = 'raw_data/stocknews/'
		file = 'numnews.txt'
		with open(path+file, 'r') as file:
			csvreader = csv.reader(file, delimiter=',')
			for row in csvreader:
				date = row[0]
				words = row[1].split(' ')
				dates.append(date)
				self.newsdata[date] = words
					
		self.mindate = min(dates)
		self.maxdate = max(dates)

	def normalize(self):
		for n in self.newsdata:
			normarray = [0 for i in range(5000)]
			array = self.newsdata[n]
			for i in array:
				if i.isdigit():
					normarray[int(i)] = 1
			self.newsdata[n] = normarray

	def getNewsData(self):
		return self.newsdata