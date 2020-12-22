import os
import pandas as pd

class SentenceGetter(object):
    
    def __init__(self, data):

        self.n_sent = 0
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["Word"].values.tolist(),s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        # print(self.grouped)
        self.sentences = [s for s in self.grouped]
        # print(self.sentences)
    
    def get_next(self):
        try:
            s = self.grouped[self.n_sent]
            self.n_sent += 1
            return s
        except:
            return None

data = pd.read_csv("dataset/ner/data_ner.csv", encoding='utf-8')
print(data.head())
getter = SentenceGetter(data)
sent = getter.get_next()
print(sent)

sentences = getter.sentences
print(len(sentences))
largest_sen = max(len(sen) for sen in sentences)
print('biggest sentence has {} words'.format(largest_sen))

"""
375
biggest sentence has 21 words
"""

