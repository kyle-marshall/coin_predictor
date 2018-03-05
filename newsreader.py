import csv
import nltk
import collections
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


'''

'''

stop_words = set(stopwords.words('english'))
numdic = {}
tokens = []
tokecount = {}
combinedtokes = []
numheadlines = []
datetokedict = {}
sortednumdates = []

with open("raw_data/stocknews/RedditNews.csv") as file:

	csvreader = csv.reader(file, delimiter=',')
	for i, row in enumerate(csvreader):
		date = row[0]
		tokenizer = RegexpTokenizer(r'[A-z]+')
		rowtokes = tokenizer.tokenize(row[1])

		filtered = [w for w in rowtokes if not w in stop_words]

		stemmer = EnglishStemmer()
		stemmed = [stemmer.stem(w) for w in filtered]

		combinedtokes.append(stemmed)
		tokens += stemmed

		if date not in datetokedict:
			datetokedict[date] = stemmed
		else:
			datetokedict[date] += stemmed

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

tokecount = collections.Counter(tokens)
commoncount = tokecount.most_common(5000)
commonwords = []

for w in commoncount:
	commonwords.append(w[0])

for i, w in enumerate(commonwords):
	numdic[w] = i

dates = []
for date in datetokedict:
	dates.append(date)

dates.sort()

for date in dates:
	sortednumdates.append((date, datetokedict[date]))

for i, pair in enumerate(sortednumdates):
	numhl = []
	for w in pair[1]:
		if w in numdic:
			numhl.append(numdic[w])
	newpair = (pair[0], set(numhl))
	sortednumdates[i] = newpair

writefile = open('raw_data/stocknews/numnews.txt', 'w')
for pair in sortednumdates:
	writefile.write(pair[0] + ',')
	for n in pair[1]:
		writefile.write(str(n) + ' ')
	writefile.write('\n')
writefile.close()