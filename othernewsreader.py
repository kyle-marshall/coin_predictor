import nltk
import csv
import gensim
import collections
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


'''

'''

stop_words = set(stopwords.words('english'))
numdic = {}
datetokedict = {}
tokens = []
tokecount = {}
combinedtokes = []
numheadlines = []

with open("raw_data/stocknews/Combined_News_DJIA.csv") as file:

	csvreader = csv.reader(file, delimiter=',')
	for i, row in enumerate(csvreader):
		date = row[0]
		combinedhl = []
		for hl in row[2:]:
			tokenizer = RegexpTokenizer(r'[A-z]+')
			rowtokes = tokenizer.tokenize(hl)

			filtered = [w for w in rowtokes if not w in stop_words]

			stemmer = EnglishStemmer()
			stemmed = [stemmer.stem(w) for w in filtered]
			
			combinedtokes.append(stemmed)
			tokens += stemmed
			combinedhl += stemmed
		if date not in datetokedict:
			datetokedict[date] = combinedhl
		else:
			datetokedict[date] += combinedhl
tokeset = set(tokens)

for i, toke in enumerate(tokeset):
	numdic[toke] = i

tokecount = collections.Counter(tokens)
commoncount = tokecount.most_common(5000)

#assign numbers to words
for date in datetokedict:
	words = datetokedict[date]
	numbers = []
	for w in words:
		numbers.append(numdic[w])
	numbers = set(numbers)
	datetokedict[date] = numbers

for date in datetokedict:
	print(date)