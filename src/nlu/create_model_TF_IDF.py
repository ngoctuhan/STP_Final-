import os
import numpy as np
from utils.pre_processing import Processor_Text
from sklearn.feature_extraction.text import TfidfVectorizer
pt = Processor_Text('dataset/stop_word.txt')

corpus = []
for filename in os.listdir('dataset/nlu'):

    filepath = os.path.join('dataset/nlu', filename)
    lines = open(filepath, 'r', encoding='utf-8').readlines()

    for line in lines:

        corpus.append(pt.normalize_sentence(line.split('\n')[0]))


print(len(corpus))
print(corpus[:10])

tf_idf = TfidfVectorizer( ngram_range= (1,1))
X = tf_idf.fit_transform(corpus)

print(len(tf_idf.get_feature_names()))

model_path = 'tf_idf_model.pickle'
import pickle
pickle.dump(tf_idf, open(model_path, 'wb'))