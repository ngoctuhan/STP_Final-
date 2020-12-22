import pandas as pd
import os
from flask import jsonify


class Intent:

    def __init__(self, dataframe_path):

        self.df = os.listdir(dataframe_path)

    def get_intent(self):
        data = []
        for name in self.df:
            path = 'dataset/nlu_answer/' + name
            f = open(path, 'r', encoding='utf-8')
            name = name.split('.')
            ans = f.read()
            data.append([name[0], ans])
            f.close()
        return data

    def get_Name(self, name):
        data = []
        for name_intent in self.df:
            if name_intent.find(name) != -1:
                path = 'dataset/nlu_answer/' + name_intent
                f = open(path, 'r', encoding='utf-8')
                ans = f.read()
                name_intent = name_intent.split('.')
                data.append([name_intent[0], ans])
                f.close()
        return data

    def sua_Intent(self, name, text):
        path = 'dataset/nlu_answer/' + name + '.txt'
        f = open(path, 'w', encoding='utf-8')
        f.write(text)
        f.close()
        pass
