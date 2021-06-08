import json
import os
import pickle
from collections import Counter

import pandas as pd


df = pd.read_csv("../data/preProcess/testData.csv")

with open("../data/english", "r") as f:
    stopWords = f.read()

    stopWordList = stopWords.splitlines()

    # 将停用词用列表的形式生成，之后查找停用词时会比较快

    stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))

review = df["review"].tolist()

reviews = [line.strip().split() for line in review]

with open("../data/wordJson/word2idx.json") as f:
    word_to_index = json.load(f)

reviewIds = [[word_to_index.get(item, word_to_index["UNK"]) for item in review] for review in reviews]

data = []

for item in reviewIds:
    if len(item) >= 200:
        data.append(item[:200])
    else:
        data.append(item + [word_to_index["PAD"]] * (200 - len(item)))

