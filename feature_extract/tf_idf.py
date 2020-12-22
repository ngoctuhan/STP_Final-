import pickle
import pandas as pd 
import os 
from utils.pre_processing import Processor_Text
MODEL_PATH  = 'models/tf_idf_model.pickle'
pt = Processor_Text('dataset/stop_word.txt')
vectorizer = pickle.load(open(MODEL_PATH,'rb'))

def extract_feature_for_test(sentence):

    global vectorizer
    global pt
    sentence = pt.normalize_sentence(sentence)

    ft = vectorizer.transform([sentence])

    return ft.toarray()

def extract_feature_for_train():

    global vectorizer
    global pt

    df_train = pd.read_csv('split_dataset/nlu/train_nlu.csv', encoding='utf-8')
    df_train = df_train.sample(frac = 1) 

    X_train = df_train['Sentence'].values
    y_train = df_train['Intent'].values

    df_test = pd.read_csv('split_dataset/nlu/test_nlu.csv', encoding='utf-8')
    df_test = df_test.sample(frac = 1) 

    X_test = df_test['Sentence'].values
    y_test = df_test['Intent'].values

    X_train = [pt.normalize_sentence(sent) for sent in X_train]
    X_test = [pt.normalize_sentence(sent) for sent in X_test]

    X_train = vectorizer.transform(X_train).toarray()
    X_test = vectorizer.transform(X_test).toarray()

    print(X_train.shape)
    print(X_test.shape)

    print(len(y_train))
    print(len(y_test))

    return X_train, y_train, X_test, y_test