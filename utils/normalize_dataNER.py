from itertools import chain
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def get_dict_map(data, token_or_tag):

    tok2idx = {}
    idx2tok = {}
    
    if token_or_tag == 'token':
        vocab = list(set(data['Word'].to_list()))
    else:
        vocab = list(set(data['Tag'].to_list()))
    
    idx2tok = {idx:tok for  idx, tok in enumerate(vocab)}
    tok2idx = {tok:idx for  idx, tok in enumerate(vocab)}
    return tok2idx, idx2tok

def get_pad_train_test_val(data_group, data, tag2idx):

    #get max token and tag length
    n_token = len(list(set(data['Word'].to_list())))
    
    # print(n_token)

    n_tag = len(list(set(data['Tag'].to_list())))
  
    #Pad tokens (X var)    
    tokens = data_group['Word_idx'].tolist()
    tokens2 = data_group['Word'].tolist()
    maxlen = max([len(s) for s in tokens])

    if maxlen < 30:
        maxlen = 30
    pad_tokens = pad_sequences(tokens, maxlen=maxlen, dtype='int32', padding='post', value= n_token)
    
    #Pad Tags (y var) and convert it into one hot encoding
    tags = data_group['Tag_idx'].tolist()
    pad_tags = pad_sequences(tags, maxlen=maxlen, dtype='int32', padding='post', value= tag2idx["O"])
    # print(pad_tags)
    n_tags = len(tag2idx)
    pad_tags = np.array([to_categorical(i, num_classes=n_tags) for i in pad_tags])

    # #Split train, test and validation set
    train_tokens, test_tokens, train_tags, test_tags = train_test_split(pad_tokens, pad_tags, test_size=0.25, train_size=0.75, random_state=2020)
    # for i in range(10): 
    #     for word, token, tag in zip(tokens2[i],pad_tokens[i], pad_tags[i]):
    #         print('%s\t%s\t%s' % (word, token, tag))
    for token, tag in zip(train_tokens[0], train_tags[0]):
        print('%s\t%s' % (token, tag))
    return train_tokens, test_tokens, train_tags, test_tags

    # return None, None, None, None


def get_data():
    
    data = pd.read_csv("dataset/ner/data_ner.csv", encoding='utf-8')
    token2idx, idx2token = get_dict_map(data, 'token')
    tag2idx, idx2tag = get_dict_map(data, 'tag')

    data['Word_idx'] = data['Word'].map(token2idx)
    data['Tag_idx'] = data['Tag'].map(tag2idx)
    # print(data.head())

    # Fill na
    data_fillna = data.fillna(method='ffill', axis=0)
    # Groupby and collect columns
    data_group = data_fillna.groupby(['Sentence #'], as_index=False)['Word', 'Tag', 'Word_idx', 'Tag_idx'].agg(lambda x: list(x))
    # Visualise data
    # print(data_group.head())
    # data_group = data_group.sample(frac = 1) 

    data_group.to_csv('data_ner_normalize.csv', index = False, encoding='utf-8')

    return get_pad_train_test_val(data_group, data, tag2idx)

