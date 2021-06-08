import pandas as pd
from bs4 import BeautifulSoup

with open("../data/rawData/unlabeledTrainData.tsv", "r", encoding='utf-8') as f:
    unlabeledTrain = [line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 2]

with open("../data/rawData/labeledTrainData.tsv", "r", encoding='utf-8') as f:
    labeledTrain = [line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 3]

with open("../data/rawData/testData.tsv", "r", encoding='utf-8') as f:
    testData = [line.strip().split("\t") for line in f.readlines() if len(line.strip().split("\t")) == 2]

unlabel = pd.DataFrame(unlabeledTrain[1:], columns=unlabeledTrain[0])  # 50000*2
label = pd.DataFrame(labeledTrain[1:], columns=labeledTrain[0])  # 25000*3

test = pd.DataFrame(testData[1:], columns=testData[0])  # 50000*2

# ==============================================数据预处理===============================================


with open("../data/english", "r") as f:

    stopWords = f.read()

    stopWordList = stopWords.splitlines()

    stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))


def cleanReview(subject):

    beau = BeautifulSoup(subject)  # 解析HTML语句

    newSubject = beau.get_text()

    newSubject = newSubject.replace("\\", "").replace("\'", "").replace('/', '').replace('"', '').replace(',',
                                                                                                          '').replace(
        '.', '').replace('?', '').replace('(', '').replace(')', '')

    newSubject = newSubject.strip().split(" ")

    newSubject = [word.lower() for word in newSubject if word not in stopWordDict]  # 所有的字母都转为小写

    newSubject = " ".join(newSubject)

    return newSubject


unlabel['review'] = unlabel['review'].apply(cleanReview)

label['review'] = label['review'].apply(cleanReview)

test['review'] = test['review'].apply(cleanReview)

label.to_csv("../data/preProcess/labeledTrainData.csv")

unlabel.to_csv("../data/preProcess/unlabeledTrainData.csv")

unlabel.to_csv("../data/preProcess/testData.csv")

# ====================================构建word2vec的语料库=========================

newDf = label['review']

newDf.to_csv('../data/preProcess/wordEmbedding.txt', index=False)

# # 将有标签的数据和无标签的数据合并，newDf只包含review
#
# newDf = pd.concat([unlabel["review"], label["review"]], axis=0)   # 75000*1
#
# # 保存成txt文件
# newDf.to_csv("../data/preProcess/wordEmbedding.txt", index=False)
