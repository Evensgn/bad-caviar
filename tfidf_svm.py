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
	
with open('stopwords.txt', encoding = 'utf-8') as f:
	buff = f.read()
	stopwords = buff.split('\n')

import jieba

data_texts_pro = []
for text in data_texts:
	text_p = text.replace('\n', '').replace('\r', '')
	words = []
	for word in jieba.cut(text_p):
		if (word not in stopwords) and (not word.isdigit()):
			words.append(word)
	text_pro = " ".join(words)
	data_texts_pro.append(text_pro)

filenames = []
for i in range(6000):
	filenames.append('comment_' + str(i))
alldata_texts = []
for filename in filenames:
	alldata_texts.append(str(read_file('data/comment_new/' + filename + '.txt')))

alldata_texts_pro = []
for text in alldata_texts:
	text_p = text.replace('\n', '').replace('\r', '')
	words = []
	for word in jieba.cut(text_p):
		if (word not in stopwords) and (not word.isdigit()):
			words.append(word)
	text_pro = " ".join(words)
	alldata_texts_pro.append(text_pro)


import numpy as np
from numpy import random

data_arr = list(zip(data_texts_pro, data_labels))
data_arr = random.permutation(data_arr)

train_texts = data_arr[:, 0]
train_labels = data_arr[:, 1].astype(int)

test_texts = np.array(alldata_texts_pro)

# test_texts = data_arr[:, 0]
# test_labels = data_arr[:, 1].astype(int)

# print(test_texts)
# print(test_labels)

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

vectorizer = TfidfVectorizer()
vectorizer.fit(test_texts)
train_texts = np.array(train_texts)
train_vecs = vectorizer.transform(train_texts)
train_vecs_arr = train_vecs.toarray()
pca = PCA(n_components = 600)
train_x = pca.fit_transform(train_vecs_arr)
# train_x = train_vecs_arr

from sklearn import svm, metrics

# clf = svm.SVR(gamma = 1)
clf = svm.SVC(gamma = 1, probability = True)
# clf = svm.SVC(kernel = 'linear', probability = True)

clf.fit(train_x, train_labels)

test_vecs = vectorizer.transform(test_texts)
test_vecs_arr = test_vecs.toarray()
test_x = pca.transform(test_vecs_arr)
# test_x = test_vecs_arr

# prediction = clf.predict(test_x)
prediction = clf.predict_proba(test_x)
# print(prediction[:, 1])

# test_auc = metrics.roc_auc_score(test_labels, prediction)
# test_auc = metrics.roc_auc_score(test_labels, prediction[:, 1])
# print(test_auc)

'''
print("Classification report for classifier %s:\n%s\n"
	  % (clf, metrics.classification_report(test_labels, prediction)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_labels, prediction))
'''

headers = ['id', 'pred']
rows = list(zip(filenames, prediction[:, 1]))
with open('ans.csv', 'w') as f:
	f_csv = csv.writer(f)
	f_csv.writerow(headers)
	f_csv.writerows(rows)
