import os
import pandas as pd
class Params:

    def __init__(self):

        self.itent2num = {}
        self.num2itent = {}
        self.itent = []
        self.extention = []
        self.__load_list_intent()
        self.weight_file = 'dataset/weight.csv'

    def __load_list_intent(self):

        for i, filename in enumerate(os.listdir('dataset/nlu')):

            x = filename.split('.')[0]
            x = x.split('_')[1]
           
            self.itent2num[x] = i
            self.num2itent[i] = x
            self.itent.append(x)

        df = pd.read_csv('dataset/nlu_extention.csv',  encoding='utf-8')

        self.extention = df['extention'].values

        self.extention = [word for word in self.extention]
        # print(self.extention)
        # print(len(self.extention))
        # stri = 'Chính sách thai sản'
        # print(stri in self.extention)


if __name__ == '__main__':

    prs = Params()

    

