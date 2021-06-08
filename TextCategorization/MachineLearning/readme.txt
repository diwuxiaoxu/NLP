环境： tensorflow == 1.15, python==3.6

test.py:
    主要实现对文本的多分类问题：
       ① 获取数据
       ② 文本预处理： 空值处理、去除stopWords、去掉特殊字符等
       ③ 特征提取：TF-IDF词向量 TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2), token_pattern='(\S+)'))
       ④ 分类器的选择