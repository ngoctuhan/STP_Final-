import pickle
import re
import os
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
import collections
import nltk
# nltk.download('punkt')

dataset = 'dataset/language_model.txt'
dataset_intent = 'dataset/nlu'

bad_chars = [';', ':', '!', "*", '.', '?',
             '!', ',', '\n', '(', ')', '...', '/', '-']
list_string = []
f = open(dataset, 'r', encoding='utf-8')
lines = f.readlines()
for line in lines:
    tmp = ''.join(i for i in line if not i in bad_chars).lower()
    list_string.append(tmp)

for filename in os.listdir(dataset_intent):
    path_file = os.path.join(dataset_intent, filename)
    lines = open(path_file, 'r', encoding='utf-8').readlines()
    for line in lines:
        tmp = ''.join(i for i in line if not i in bad_chars).lower()
        list_string.append(tmp)

# -------------------------------------------------------------

queue = collections.deque(maxlen=3)
vocal = set()
unigram = {}
bigram = {}
trigram = {}

# ---------------------------------------------------------------
for line in list_string:
    tokens = word_tokenize(line)
    tokens.append('')

    for token in tokens:
        queue.append(token)

        if token not in vocal:
            vocal.add(token)

        if token not in unigram:
            unigram[token] = 0
        unigram[token] += 1

        if len(queue) >= 2:
            item = tuple(queue)[:2]

            if item not in bigram:
                bigram[item] = 0
            bigram[item] += 1

        if len(queue) == 3:
            item = tuple(queue)

            if item not in trigram and token != '':
                trigram[item] = 0

            if token != '':
                trigram[item] += 1

total_word = len(unigram)
unigram[''] = total_word

# save model


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


save_obj(unigram, "unigram")

save_obj(bigram, "bigram")

save_obj(trigram, "trigram")

save_obj(vocal, 'vocal')
