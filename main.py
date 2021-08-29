import os
import csv
import time
import argparse
import numpy as np
import pandas as pd
import pysentiment2 as ps
import vaderSentiment.vaderSentiment as vs

LEXICON = ('vader', 'LM', 'update', 'synthesis')

# parse file path and store it.
class Filepath(object):

	def __init__(self, filepath):
		self.filepath = filepath
		self.filepath = self.filepath.replace('\\', '/').replace('//', '/')
		self._parse_path_filename()
		self._parse_name_suffix()

	# parse path and filename
	def _parse_path_filename(self):
		parts = self.filepath.split('/')
		if len(parts) < 2:
			self.path = ''
			self.filename = parts[0]
		else:
			self.path = '/'.join(parts[:-1])
			self.filename = parts[-1]

	# parse name and suffix
	def _parse_name_suffix(self):
		parts = self.filename.split('.')
		if len(parts) < 2:
			self.name = parts[0]
			self.suffix = ''
		else:
			self.name = parts[0]
			self.suffix = '.'.join(parts[1:])

	# generate target file path with specify flag
	def add_flag(self, flag):
		target = ''
		if self.path != '':
			target = self.path + '/'
		target += self.name
		target += '_' + flag

		if self.suffix != '':
			target += '.' + self.suffix
		return target

# instance lexicon
def create_lexicon(name):
	if name == 'vader':
		return vs.SentimentIntensityAnalyzer()
	if name == 'LM':
		return ps.LM()
	if name == 'synthesis':
		return ps.SYNTHESIS()
	if name == 'update':
		return vs.SentimentIntensityAnalyzer('synthesis_lexicon.txt')


# process the .csv file with specify lexicon and store the result
def handle(args):
	src = Filepath(args.file)
	src.path = args.output
	if src.suffix != 'csv':
		print('Not handle such file: \'%s\'' % src.filepath)
		return

	reader = pd.read_csv(src.filepath, usecols=['uid', 'date', 'message']).fillna('')
	models = [create_lexicon(name) for name in args.lexicon]

	datas = dict()
	for index, row in reader.iterrows():
		uid = row['uid']
		date = row['date'].split(' ')[0]
		text = row['message']

		if uid not in datas:
			datas[uid] = dict()

		if date not in datas[uid]:
			datas[uid][date] = dict()
			for name in args.lexicon:
				datas[uid][date][name] = list()

		for i in range(len(models)):
			name = args.lexicon[i]
			score = models[i].score(text)
			datas[uid][date][name].append(score)
	
	src.name = 'chat'
	for uid in datas:
		result = dict()
		for name in args.lexicon:
			result[name] = dict()
			for date in datas[uid]:
				result[name][date] = np.mean(datas[uid][date][name])
		writer = pd.DataFrame(result)
		writer.to_csv(src.add_flag(uid))

if __name__ == '__main__':
	# parse the argument
	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--lexicon', nargs='+', choices=LEXICON)
	group = parser.add_mutually_exclusive_group()
	group.add_argument('-f', '--files', nargs='+')
	group.add_argument('-d', '--dir')
	parser.add_argument('-o', '--output', default='./output')
	parser.add_argument('--detail', action='store_true')
	args = parser.parse_args()


	if not os.path.exists(args.output):
		os.mkdir(args.output)

	# distribute the tasks
	if args.lexicon and len(args.lexicon) > 0:
		if args.files:
			for file in args.files:
				args.file = file
				handle(args)
			
		if args.dir:
			for root, dirs, files in os.walk(args.dir):
				for file in files:
					args.file = '%s/%s' % (root, file)
					handle(args)
				break
