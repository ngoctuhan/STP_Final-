import pandas as pd


class CV:

    def __init__(self, dataframe_path):

        self.df = pd.read_csv(dataframe_path, encoding="utf-8").values

    def get_cv(self):
        return self.df

    def get_Designation(self, designation):
        tmp = pd.read_csv('dataset/infor_cv/cv.csv')

        # '''tmp1 = tmp[str(tmp['Designation']).find(designation)!=-1]
        # print(tmp1)'''


if __name__ == "__main__":
    cv = CV('utils/cv.csv')
    print(cv.get_cv())
