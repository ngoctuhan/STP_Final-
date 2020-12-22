import numpy as np
import os
import random
import pandas as pd
import matplotlib.pyplot as plt

data_nlu = 'dataset/nlu'

test_size = 0.15
X_train, y_train, X_test, y_test = [], [], [], []

nb_train, nb_test, lb = [], [], []

for filename in os.listdir(data_nlu):

    pathfile = os.path.join(data_nlu, filename)
    lines = open(pathfile, 'r', encoding='utf-8').readlines()
    random.shuffle(lines)

    split_value = int(len(lines) * (1-test_size))
    train =  lines[:split_value]
    test  =  lines[split_value:]
    label = filename.split('.txt')[0]
    label = label.split('_')[1]

    # append to list
    X_train += train
    y_train += [label] * len(train)
    X_test  += test 
    y_test  += [label] * len(test)

    nb_train.append(len(train))
    nb_test.append(len(test))
    lb.append(label[:3])


def draw_distribute(list_nb, list_lb):
    fig, axes = plt.subplots()
    plt.bar(list_lb,list_nb)

    # Add labels
    plt.title('Distribution of dataset')
    plt.xlabel('Words')
    plt.ylabel('Number samples')
    plt.show()

# draw_distribute(nb_train, lb)
# draw_distribute(nb_test, lb)

X_train = np.array(X_train).reshape(-1, 1)
X_test = np.array(X_test).reshape(-1, 1)
y_train = np.array(y_train).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)
# save to csv    
merge_train = np.concatenate((X_train, y_train), axis =  1)
merge_test = np.concatenate((X_test, y_test), axis =  1)
columns = ['Sentence', 'Intent']

train_file = os.path.join('split_dataset/nlu', 'train_nlu.csv')
test_file = os.path.join('split_dataset/nlu', 'test_nlu.csv')
df_train = pd.DataFrame(merge_train, columns=columns)
df_train.to_csv(train_file, index=False, encoding='utf-8')

df_test = pd.DataFrame(merge_test, columns=columns)
df_test.to_csv(test_file, index=False, encoding='utf-8')
