import pandas as pd


class CV:

    def __init__(self, dataframe_path):

        self.df = pd.read_csv(dataframe_path, encoding="utf-8").values

    def get_cv(self):
        return self.df

    def search_cv(self, filter, folder_data):

        filter = filter.lower()
        if folder_data is None:
            folder_data = 'dataset/infor_cv/cv.csv'

        df = pd.read_csv(folder_data)

        # result = df[filter in df['Designation']]

        tmp = [str(v).lower().replace("\n", " ")
               for v in df['Designation'].values]

        result = [i for i, des in enumerate(tmp) if des.find(filter) >= 0]

        X = df.values[result]
        X = X.tolist()
        return X
