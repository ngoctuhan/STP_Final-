import os
import numpy as np
from underthesea import word_tokenize
from pre_processing import Processor_Text
# create dictionary for BoW or TF-IDF

p = Processor_Text('stop_word.txt')
words = []
for file in os.listdir('nlu'):

    file_path = os.path.join('nlu', file)

    f = open(file_path, 'r', encoding='utf-8').readlines()

    for sentence in f:
        x = word_tokenize(sentence)
        x =  p.remove_stop_word(x)
        print(x)
        sentence = p.synonyymous(x)
        words.append(sentence)
    

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle
vectorizer = CountVectorizer(decode_error="replace")
vec_train = vectorizer.fit_transform(words)
#Save vectorizer.vocabulary_
pickle.dump(vectorizer.vocabulary_,open("tf-idf.pkl","wb"))

#Load it later
transformer = TfidfTransformer()
loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("tf-idf.pkl", "rb")))
tfidf = transformer.fit_transform(loaded_vec.fit_transform(np.array(["Tôi muốn vào công ty làm trong tháng này"])))

print(tfidf)