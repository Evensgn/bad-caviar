import csv

def read_file(path):
    with open(path, 'r', encoding = 'utf-8') as f:
        content = f.read()
        return content

with open('data/training_new.csv') as f:
	f_csv = csv.DictReader(f)
	data_texts = []
	data_labels = []
	for row in f_csv:
		data_texts.append(str(read_file('data/comment_new/' + row['id'] + '.txt')))
		data_labels.append(int(row['pred']))
	
with open('stopwords.txt') as f:
	buff = f.read()
	stopwords = buff.split('\n')

import jieba

data_texts_pro = []
for text in data_texts:
	text_p = text.replace(' ', '').replace('\n', '').replace('\r', '')
	words = []
	for word in jieba.cut(text_p):
		if (word not in stopwords) and (not word.isdigit()):
			words.append(word)
	text_pro = " ".join(words)
	data_texts_pro.append(text_pro)

train_texts = data_texts_pro[:800]
train_labels = data_labels[:800]
test_texts = data_texts_pro[800:]
test_labels = data_labels[800:]

import numpy as np
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

vectorizer = TfidfVectorizer()
train_texts = np.array(train_texts)
train_vecs = vectorizer.fit_transform(train_texts)
train_vecs_arr = train_vecs.toarray()
pca = PCA(n_components = 100)
train_x = pca.fit_transform(train_vecs_arr)

from sklearn import svm, metrics

clf = svm.SVC(gamma = 1)
# clf = svm.SVC(kernel = 'linear')

clf.fit(train_x, train_labels)

test_vecs = vectorizer.transform(test_texts)
test_vecs_arr = test_vecs.toarray()
test_x = pca.transform(test_vecs_arr)

prediction = clf.predict(test_x)

print("Classification report for classifier %s:\n%s\n"
	  % (clf, metrics.classification_report(test_labels, prediction)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, prediction))
