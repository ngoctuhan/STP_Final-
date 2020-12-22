import pandas as pd


class Login:

    def __init__(self, dataframe_path):

        self.df = pd.read_csv(dataframe_path, encoding="utf-8").values

    def check_login(self, username, password):

        if username in self.df[:, 0] and password in self.df[:, 1]:
            return True

        else:
            return False

    def check_user(self, username):
        if username in self.df[:, 0]:
            return True
        else:
            return False

    def check_email(self, email):
        if email in self.df[:, 2]:
            return True
        else:
            return False

    def get_status(self, username):
        return 0

    def create_acount(self, username, password, email):
        tmpp = pd.read_csv('dataset/account.csv', encoding="utf-8")
        tmpp = tmpp.append({'username': username, 'password': password,
                            'email': email, 'status': int(1)}, ignore_index=True)
        tmpp.to_csv('dataset/account.csv', index=False)
        pass
