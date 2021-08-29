import os
import numpy as np
import pandas as pd
import pysentiment2 as ps
import vaderSentiment.vaderSentiment as vs

def open_vader_lexicon(filename):
	f = open(filename, 'r', encoding='utf-8')
	lines = f.read().split('\n')
	f.close()

	lexicon = dict()

	for line in lines:
		if line == '': continue
		parts = line.split('\t')
		lexicon[parts[0].lower()] = float(parts[1])

	return lexicon

def open_lm_lexicon(filename):
	reader = pd.read_csv(filename, usecols=['Word', 'Positive', 'Negative']).fillna('')

	lexicon = dict()

	for index, row in reader.iterrows():
		if row['Word'] == '': continue
		if row['Positive'] > 0:
			lexicon[row['Word'].lower()] = 1
		if row['Negative'] > 0:
			lexicon[row['Word'].lower()] = -1

	return lexicon	



if __name__ == '__main__':
	vader_lexicon = open_vader_lexicon('./vaderSentiment/vader_lexicon.txt')
	print('vader lexicon: ', len(vader_lexicon))
	lm_lexicon = open_lm_lexicon('./pysentiment2/static/LM.csv')
	print('LM lexicon: ', len(lm_lexicon))

	lexicon = {**lm_lexicon, **vader_lexicon}

	print('vader + LM lexicon: ', len(lexicon))

	vader = vs.SentimentIntensityAnalyzer()

	datas = dict()

	for root, dirs, files in os.walk('./data'):
		for file in files:
			filepath = '%s/%s' % (root, file)
			reader = pd.read_csv(filepath, usecols=['message',]).fillna('')
			for index, row in reader.iterrows():
				words = vader.tokenize(row['message'])
				for word in words:
					word_low = word.lower()
					if not word_low.encode('UTF-8').isalpha(): continue
					if word_low in lexicon: continue
					if word_low in datas:
						datas[word_low] += 1
					else:
						datas[word_low] = 1

	result = {
		'word': list(),
		'frequence': list()
	}
	for k, v in datas.items():
		result['word'].append(k)
		result['frequence'].append(v)

	writer = pd.DataFrame(result)
	writer.to_csv('./statics.csv', index=0)






