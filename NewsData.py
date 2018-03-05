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

	def getNewsData():
		return self.newsdata