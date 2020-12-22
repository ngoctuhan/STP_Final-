import pandas as pd


class Log:

    def __init__(self, dataframe_path):

        self.df = pd.read_csv(dataframe_path,encoding="utf-8").values
    def get_log(self):
        return self.df
if __name__ == "__main__":
    log = Log('dataset/log_chat.csv')
    print(log.get_log())