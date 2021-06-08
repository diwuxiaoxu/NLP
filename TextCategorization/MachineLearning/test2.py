from ast import literal_eval
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
import nltk
import re
from nltk.corpus import stopwords

# ======获取数据======

# 下载stopWord,一些没有意义的词语
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))


# literal_eval 检查文本的合法性

def read_data(filename):
    data = pd.read_csv(filename, sep='\t')

    data['tags'] = data['tags'].apply(literal_eval)

    return data


train = read_data('../data/Multil_label_data/train.tsv')  # 100000条数据

validation = read_data('../data/Multil_label_data/validation.tsv')  # 30000条数据


# ========文本预处理=========

# 转化tag

tags = train['tags'].values

# tag_dic: {tag:count}

tag_dic = {}

for tag_list in tags:

    for tag in tag_list:

        if tag not in tag_dic:

            tag_dic[tag] = 1

        else:

            tag_dic[tag] += 1

# 因为是英文，所以不存在分词

# 用空格替换各种符号

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

# 删除各种符号

BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

STOPWORDS = set(stopwords.words('english'))


# 定义数据清洗函数

def text_prepare(text):
    text = text.lower()  # 字母小写化

    text = REPLACE_BY_SPACE_RE.sub(' ', text)

    text = BAD_SYMBOLS_RE.sub('', text)

    text = ' '.join([w for w in text.split() if w not in STOPWORDS])  # 删除停用词

    return text


X_train, y_train = train.title, train.tags

X_val, y_val = validation.title, validation.tags

# 开始进行数据清洗

X_train = [text_prepare(x) for x in X_train]

X_val = [text_prepare(x) for x in X_val]

# =======特征提取======

# tfidf = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern='(\S+)')
#
# feature = tfidf.fit_transform(X_train)

# 生成多标签的词袋矩阵
# 每个样本的tag是这样的 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

mlb = MultiLabelBinarizer(classes=sorted(tag_dic.keys()))

y_train = mlb.fit_transform(y_train)

y_val = mlb.fit_transform(y_val)

# =======分类器的选择======
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

LogReg_pipeline = Pipeline([

    ('tfidf', TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern='(\S+)')),

    ('clf', OneVsRestClassifier(LogisticRegression(), n_jobs=1)),

])

LogReg_pipeline.fit(X_train, y_train)

y_pred = LogReg_pipeline.predict(X_val)

from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score

print('accuracy %s' % accuracy_score(y_pred, y_val))
print('precision %s' % precision_score(y_pred, y_val, average='macro'))
print('recall %s' % recall_score(y_pred, y_val, average='macro'))
print('f1 %s' % f1_score(y_pred, y_val, average='macro'))
