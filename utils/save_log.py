import os
import pandas as pd
from datetime import date
from datetime import datetime
import numpy as np


def save_log(file_log, question, answer, ip):

    columns = ['question', 'answer', 'sender', 'time']
    today = date.today()
    d4 = today.strftime("%b-%d-%Y")

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    time = d4 + " | " + current_time

    data = [question, answer, ip, time]

    if file_log is None or os.path.exists(file_log) == False:
        data = np.array(data).reshape(1, -1)
    else:
        df = pd.read_csv(file_log, encoding='utf-8').values
        data = np.array(data).reshape(1, -1)
        data = np.concatenate((df, data), axis=0)

    df_save = pd.DataFrame(data=data, columns=columns)
    df_save.to_csv(file_log, index=False)
