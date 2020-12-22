# mapping from a test to label
import numpy as np
import pickle
from feature_extract.tf_idf import extract_feature_for_test
import time
import os
import pandas as pd
import ast


class Selena:

    def __init__(self, model_path):

        self.model_path = model_path
        self.MAPPING = {}
        self.__load_model()
        self.__load_answer()
        self.__load_usually()

    def __load_model(self):

        self.model = pickle.load(open(self.model_path, 'rb'))

    def __load_answer(self):

        answer_path = 'dataset/nlu_answer'
        for filename in os.listdir(answer_path):
            key = filename.split('.')[0]
            key = key.split('_')[1]
            path_file = os.path.join('dataset/nlu_answer', filename)
            self.MAPPING[key] = open(path_file, 'r', encoding='utf-8').read()

    def __load_usually(self):

        path = 'dataset/nlu_extention.csv'
        self.df = pd.read_csv(path, encoding='utf-8')
        self.df2 = pd.read_csv('dataset/nlu_details.csv', encoding='utf-8')
        # self.dataframe = df
        # intent_group = df.groupby('intent')
        # print(intent_group.head())

    def predict(self, sentence):

        # t1 =  time.time()
        X_test = extract_feature_for_test(sentence)
        # t2 = time.time()
        predict = self.model.predict(X_test)
        # t3 = time.time()

        prob = max(self.model.predict_proba(X_test)[0])
        return predict[0], self.MAPPING[predict[0]], prob * 100

    def answer(self, intent):
        return self.MAPPING[intent]

    def relation_answer(self, intent):

        result = self.df.where(self.df["intent"] == intent, inplace=False)
        result = result.dropna()

        return result['extention'].values

    def details_extention(self, extention):
        result = self.df.where(
            self.df["extention"] == extention, inplace=False)
        result = result.dropna()
        string = result['details'].values[0]
        return ast.literal_eval(string)

    def answer_details(self, question):

        result = self.df2.where(
            self.df2["question"] == question, inplace=False)
        result = result.dropna()

        return result['answer'].values

    def usually_answer_next(self, intent):

        pass
