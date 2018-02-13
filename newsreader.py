import nltk
import csv
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


'''

'''

stop_words = set(stopwords.words('english'))
combinedtokes = []

with open("RedditNews.csv") as file:
	csvreader = csv.reader(file, delimiter=',')
	for row in csvreader:
		tokenizer = RegexpTokenizer(r'\w+')
		rowtokes = tokenizer.tokenize(row[1])
		filtered = [w for w in rowtokes if not w in stop_words]

		stemmer = EnglishStemmer()
		stemmed = [stemmer.stem(w) for w in filtered]
		print(stemmed)
		combinedtokes += stemmed
	

