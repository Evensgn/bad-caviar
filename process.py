import csv

def read_file(path):
    with open(path, 'r', encoding = 'utf-8') as f:
        content = f.read()
        return content

with open('data/training_new.csv') as f:
	f_csv = csv.DictReader(f)
	train_texts = []
	train_labels = []
	for row in f_csv:
		train_texts.append(str(read_file('data/comment_new/' + row['id'] + '.txt')))
		train_labels.append(int(row['pred']))
	
with open('stopwords.txt') as f:
	buff = f.read()
	stopwords = buff.split('\n')
print(stopwords)

import jieba

for text in train_texts:
	text_p = text.replace(' ', '').replace('\n', '').replace('\r', '')
	words = []
	for word in jieba.cut(text_p):
		if (word not in stopwords) and (not word.isdigit()):
			words.append(word)
	print(words) 
