import pandas as pd

import numpy as np

import jieba as jb

import re

from sklearn.feature_extraction.text import TfidfVectorizer

# ======获取数据======

df = pd.read_csv('../data/shopping.csv')

df = df[['cat', 'review']]

# df(5*2)
#        cat                                             review
# 8896    平板  我只想说客服，你要买东西的时候，随时联系的到，买完了，问他们为什么不送东西的时候，你可能就被...
# 33299  洗发水                                 感觉一般还不如海飞丝呢，洗完头发很干


# ======数据处理=======

# 清洗数据中的空值

df = df[pd.notnull(df['review'])]

# 多分类问题，cat转化为id，方便后面的分类

df['cat_id'] = df['cat'].factorize()[0]

cat_id_df = df[['cat', 'cat_id']].drop_duplicates().sort_values('cat_id').reset_index(drop=True)

# cat_to_id {'书籍': 0, '平板': 1, '手机': 2, '水果': 3, '洗发水': 4, '热水器': 5, '蒙牛': 6, '衣服': 7, '计算机': 8, '酒店': 9}

cat_to_id = dict(cat_id_df.values)

id_to_cat = dict(cat_id_df[['cat_id', 'cat']].values)


# df (5*3)
#        cat                                             review  cat_id
# 19939   水果                  京东快递就是快，昨天晚上下的单，今天早上就到了，也不酸，苹果好脆，       3
# 8830    平板                               京东就是快，棒棒哒，一如既往的好，给力！       1
# 32590  洗发水  明明下单了两套礼盒，结果今天只送货送到了一套？！是快递人员的疏忽还是厂家的失误？！还是说这就...       4
# 45829   衣服                                            码偏小，不能穿       7
# 55862   酒店  每次来都住这里，交通很方便，房间也宽敞。向大家推荐海景的房间，能看到深港跨海大桥和对面的香港...       9

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

# df (5*4)
#        cat  ...                                       clean_review
# 52007  计算机  ...                 键盘不是很好上下左右键有点歪还有就是VISTA需要适应很累要好好研究
# 18103   水果  ...     苹果很甜大部分很新鲜有小部分苹果可能是运输途中磕碰较厉害联系京东客服已妥善解决为京东客服点赞
# 28072  洗发水  ...                           今天下单今天下午就到货了还没有用不知道效果怎么样
# 50221  计算机  ...  之前没也没去商场看没想那样轻巧很喜欢换了XP速度很不错普通的使用绝对没有问题独立的128显卡...
# 18850   水果  ...                                  买过多次了一如既往的好吃非常之满意

# 分词

stopwords = stopwordslist("../data/chineseStopWords.txt")

df['cut_review'] = df['clean_review'].apply(lambda x: " ".join([w for w in list(jb.cut(x)) if w not in stopwords]))

# df (5*5)
#       cat  ...                                         cut_review
# 42461  衣服  ...  帮 男朋友 买 质量 真的 超级 款式 好看 对象 喜欢 以后 买 裤子 你家 真的 喜欢 ...
# 7863   平板  ...               没有 送 评论 说 钢化 膜 保护套 拿到 手 没有 太 地道 太小 气
# 14881  手机  ...  机子 外形 不错 外屏 容易 按上 手印 内屏 万 千色 不了 26 万 在我看来 颜色 漂...
# 1720   书籍  ...  书中 很多 内容 骇人听闻 史料 机密 程度 怀疑 真实性 作者 有着 特殊 行业 背景 细...
# 48750  衣服  ...  东西 真心 穿 真心 显 老气 质量 不好 客服 真是 知道 改 说发 信息 问 物流 信息...

# =======特征提取======

tfidf = TfidfVectorizer()

# 可以扩充特征数量：tfidf = TfidfVectorizer(norm='l2', ngram_range=(1, 2))
# ngram_range=(1,2)代表不仅仅是单个词的TF-IDF，也包括每个词与它相邻词组成一个新的词语，使得总体样本数量增加

features = tfidf.fit_transform(df.cut_review)

labels = df.cat_id

# features (62773, 67883)
# 62773表示所有的样本数据，67883表示特征数量这里代表所有词语数
# features

# ========分类器的选择======

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

import warnings

warnings.filterwarnings("ignore")

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.33, random_state=0)

# 1.生成词频向量. 2.生成TF-IDF向量

# count_vect = CountVectorizer()

# X_train_counts = count_vect.fit_transform(X_train)

# tfidf_transformer = TfidfTransformer()

# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# 朴素贝叶斯分类器

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score

print('accuracy %s' % accuracy_score(y_pred, y_test))
print('precision %s' % precision_score(y_pred, y_test, average='macro'))
print('recall %s' % recall_score(y_pred, y_test, average='macro'))
print('f1 %s' % f1_score(y_pred, y_test, average='macro'))


# 朴素贝叶斯
# accuracy 0.8171461672137478
# precision 0.6698859940856876
# recall 0.7959108631062619
# f1 0.7034863030937915

# 逻辑回归

# from sklearn.linear_model import LogisticRegression
#
# model = LogisticRegression()
#
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
#
# from sklearn.metrics import recall_score, accuracy_score, f1_score, precision_score
#
# print('accuracy %s' % accuracy_score(y_pred, y_test))
# print('precision %s' % precision_score(y_pred, y_test, average='macro'))
# print('recall %s' % recall_score(y_pred, y_test, average='macro'))
# print('f1 %s' % f1_score(y_pred, y_test, average='macro'))

# LR
# accuracy 0.8651284031666345
# precision 0.8252060675345074
# recall 0.9024077569511062
# f1 0.8514898534744866


# ========定义预测函数======


def myPredict(sec):

    format_sec = " ".join([w for w in list(jb.cut(remove_punctuation(sec))) if w not in stopwords])

    # 转化为词频向量

    count_vector = CountVectorizer()

    count_vector.fit(df['cut_review'])

    pred_cat_id = model.predict(count_vector.transform([format_sec]))

    print(id_to_cat[pred_cat_id[0]])


myPredict('都烂了，下次再也不买了')
