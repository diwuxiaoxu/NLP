import pandas as pd

import numpy as np

import jieba as jb

import re

# ======获取数据======
from keras import Sequential
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

df = pd.read_csv('../data/shopping.csv')

df = df[['cat', 'review']]


# ======数据处理=======

# 清洗数据中的空值

df = df[pd.notnull(df['review'])]

df['cat_id'] = df['cat'].factorize()[0]

cat_id_df = df[['cat', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)

# cat_to_id {'书籍': 0, '平板': 1, '手机': 2, '水果': 3, '洗发水': 4, '热水器': 5, '蒙牛': 6, '衣服': 7, '计算机': 8, '酒店': 9}

cat_to_id = dict(cat_id_df.values)

id_to_cat = dict(cat_id_df[['cat_id', 'cat']].values)

# 删除除了字母，数字，汉字以外的其它符号

def remove_punctuation(line):
    line = str(line)
    if line.strip() == '':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('', line)
    return line


def stopwordslist(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords


df['clean_review'] = df['review'].apply(remove_punctuation)

stopwords = stopwordslist("../data/chineseStopWords.txt")

# 分词
df['cut_review'] = df['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))

# =====================================================LSTM建模========================================================

# 在机器学习中，我们使用TF-IDF进行特征提取；在深度学习中，我们使用keras的分词器Tokenizer + word Embedding

# num_words: 返回使用最频繁的词语个数

tokenizer = Tokenizer(num_words=50000)

tokenizer.fit_on_texts(df['cut_review'].values)  # 通过tokenizer分词器建立出 word:index 之间的关系

# texts_to_sequence(texts): text是待转为序列的文本列表

X = tokenizer.texts_to_sequences(df['cut_review'].values)

# 填充X,让X的各个行的长度统一

X = pad_sequences(X, maxlen=250)

# 多类标签的one-hot展开

Y = pd.get_dummies(df['cat_id']).values


# 拆分训练集和测试集

X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.10, random_state=42)


# 定义模型

model = Sequential()

model.add(Embedding(input_dim=50000, output_dim=100, input_length=X.shape[1]))

model.add(SpatialDropout1D(0.2))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

history = model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=2)

y_pred = model.predict(X_test)

y_pred = np.rint(y_pred)

from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score

print('accuracy %s' % accuracy_score(y_pred, y_test))
print('precision %s' % precision_score(y_pred, y_test, average='macro'))
print('recall %s' % recall_score(y_pred, y_test, average='macro'))
print('f1 %s' % f1_score(y_pred, y_test, average='macro'))


def predict(text):

    txt = remove_punctuation(text)

    txt = [" ".join([w for w in list(jb.cut(txt)) if w not in stopwords])]

    seq = tokenizer.texts_to_sequences(txt)

    padded = pad_sequences(seq, maxlen=250)

    pred = model.predict(padded)

    cat_id= pred.argmax(axis=1)[0]

    return cat_id_df[cat_id_df.cat_id==cat_id]['cat'].values[0]


