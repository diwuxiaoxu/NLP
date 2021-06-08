import logging

import pandas
from gensim.models import word2vec, Word2Vec, KeyedVectors

# 设置输出日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 直接用gemsim提供的API去读取txt文件，读取文件的API有LineSentence 和 Text8Corpus, PathLineSentences等。

sentences = word2vec.LineSentence("../data/preProcess/wordEmbedding.txt")

df = pandas.read_csv("../data/preProcess/wordEmbedding.txt")


# 训练模型，词向量的长度设置为100， 迭代次数为5，采用skip-gram模型，模型保存为bin格式
model = Word2Vec(sentences, vector_size=100, sg=1, epochs=5)  # Word2Vec(vocab=58733, vector_size=100, alpha=0.025)

# model.wv.save_word2vec_format("./word2Vec.bin" , binary=True)

model.save('word2vec.model')

