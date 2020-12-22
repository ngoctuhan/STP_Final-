# covert data NER to CSV file
# ! pip install underthesea
from underthesea import word_tokenize
import pandas as pd
import numpy as np
import os

file_name = 'NER_DATASET.txt'
lines = open(file_name, 'r', encoding= 'utf-8').readlines()

sentence_ID, tu , tag = [], [], []
for i, line in enumerate(lines):

    line = line.split("/n")[0].lower()
    sentence = word_tokenize(line)
    new_sentence = []
    num_dong = 0
    num_mo   = 0 
    cluster = ''
    for ii in range(len(sentence)):
        sentence[ii] = sentence[ii].replace(" ", "_")
    print(sentence)
    for j, word in enumerate(sentence):

        if word == '<' and num_mo == 0:
            num_mo += 1

        elif num_dong == 0 and num_mo == 0:
            sentence_ID.append(str(i))
            tu.append(word)
            tag.append('O')

        elif word == '>' and num_mo == 1 and num_dong == 0:
          num_dong += 1
          tag.append('E-' + sentence[j-1])

        elif word == '<' and num_dong == 1:
          sentence_ID.append(str(i))
          num_mo += 1
          tu.append(cluster)
          cluster = ''

        elif num_mo == 1 and num_dong ==  1:
          
          if sentence[j+1] != '<':
            cluster += word + '_'
          else:
            cluster += word

        elif word == '>' and num_mo == 2:

          num_mo = 0
          num_dong = 0
          cluster = ''
      
  
print(len(sentence_ID))
print(len(tu))
print(len(tag))  

colum = ['Sentence #', 'Word', 'Tag']
import numpy as np

sentence_ID = np.array(sentence_ID).reshape(-1,1)
tu =  np.array(tu).reshape(-1, 1)
tag = np.array(tag).reshape(-1, 1)


X =  np.concatenate((sentence_ID, tu, tag), axis = 1)

df = pd.DataFrame(X, columns=colum)

print(df.head())

df.to_csv('data_ner.csv', index = False, encoding='utf-8')