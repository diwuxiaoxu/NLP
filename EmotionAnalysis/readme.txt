基于IMDB电影评论的情感分析数据集 /data/rawData
    数据集：带标签的训练数据集：labeledTrainData.tsv
            不带标签的训练数据集：unlabeledTrainData.tsv
            测试集：testData.tsv
     字段的含义：
        id: 电影评论的id
        review：电影评论的内容
        sentiment：情感分类的标签 1/0


一、数据预处理

     将原始数据处理成干净的数据，去除各种标点符号  /dataHelper/processData.py

     生成训练word2vec模型的输入数据 /data/preProcess/wordEmbedding.txt

二、训练word2vec

      预训练word2vec词向量  /word2vec/genWord2Vec.py

       预训练的词向量保存为bin格式 /word2vec/Word2Vec.bin

三、 Transformer

