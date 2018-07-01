import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import sys
import re
import nltk
import nltk.stem
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.layers import Dense, Activation, Flatten, Convolution2D, Dropout, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.models import Sequential
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
import csv

def clean_text(comment_text):
    comment_list = []
    for text in comment_text:
        # 将单词转换为小写
        text = text.lower()
        # 删除非字母、数字字符
        text = re.sub(r"[^a-z']", " ", text)
        # 恢复常见的简写
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"cannot", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"\'m", " am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " will ", text)
        text = re.sub(r"ain\'t", " are not ", text)
        text = re.sub(r"aren't", " are not ", text)
        text = re.sub(r"couldn\'t", " can not ", text)
        text = re.sub(r"didn't", " do not ", text)
        text = re.sub(r"doesn't", " do not ", text)
        text = re.sub(r"don't", " do not ", text)
        text = re.sub(r"hadn't", " have not ", text)
        text = re.sub(r"hasn't", " have not ", text)
        text = re.sub(r"\'ll", " will ", text)
        # 换掉一些停词，貌似更差了
        # text = re.sub(r"am", "", text)
        #进行词干提取
        new_text = ""
        s = nltk.stem.snowball.EnglishStemmer()  # 英文词干提取器
        for word in word_tokenize(text):
            new_text = new_text + " " + s.stem(word)
        # 放回去
        comment_list.append(new_text)
    return comment_list

def read_data(file):
    train_data = csv.reader(open(file, encoding="utf-8"))
    lines = 0
    for r in train_data:
        lines += 1
    train_data_label = np.zeros([lines - 1, ])
    train_data_content = []
    train_data = csv.reader(open(file, encoding="utf-8"))
    i = 0
    for data in train_data:
        if data[0] == "Label" or data[0] == "SmsId":
            continue
        if data[0] == "ham":
            train_data_label[i] = 0
        if data[0] == "spam":
            train_data_label[i] = 1
        train_data_content.append(data[1])
        i += 1
    print(train_data_label.shape, len(train_data_content))
    return train_data_label,train_data_content


# 载入数据
train_y,train_data_content = read_data("./垃圾短信分类data/train.csv")
_,test_data_content = read_data("./垃圾短信分类data/test.csv")
train_data_content = clean_text(train_data_content)
test_data_content = clean_text(test_data_content)

# 数据的TF-IDF信息计算
all_comment_list = list(train_data_content) + list(test_data_content)
text_vector = TfidfVectorizer(sublinear_tf=True, strip_accents='unicode',token_pattern=r'\w{1,}',
                              max_features=5000, ngram_range=(1, 1), analyzer='word')
text_vector.fit(all_comment_list)
train_x = text_vector.transform(train_data_content)
test_x = text_vector.transform(test_data_content)
train_x = train_x.toarray()
test_x = test_x.toarray()
print(train_x.shape,test_x.shape,type(train_x))

# 构建模型
clf = LogisticRegression(C=100.0)
clf.fit(train_x, train_y)
train_scores = clf.score(train_x, train_y)
print(train_scores)
test_y = clf.predict_proba(test_x)

# 预测答案
print(test_y.shape)
answer = pd.read_csv(open("./垃圾短信分类data/sampleSubmission.csv"))
for i in range(test_y.shape[0]):
    predit = test_y[i,0]
    if predit < 0.5:
        answer.loc[i,"Label"] = "spam"
    else:
        answer.loc[i,"Label"] = "ham"
answer.to_csv("./垃圾短信分类data/submission.csv",index=False)  # 不要保存引索列
