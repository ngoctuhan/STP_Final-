import numpy as np
import ast 
import tensorflow
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from tensorflow.keras.utils import plot_model
from normalize_dataNER import get_data
import pandas as pd
from utils.metric import f1_m, precision_m, recall_m
from numpy.random import seed
import tensorflow.keras as k
seed(1)
# tensorflow.random.set_seed(2)

datax = pd.read_csv("dataset/ner/data_ner.csv", encoding='utf-8')
# data_groupx = pd.read_csv("data_ner_normalize.csv", encoding='utf-8')
# print(data_group.head())

input_dim = len(list(set(datax['Word'].to_list())))+1
output_dim = 64
# tokens = data_group['Word_idx'].values

# tokens = [ast.literal_eval(token) for token in tokens ]

# input_length = max(len(token) for token in tokens)

# if input_length < 30:
    # input_length =  30


input_length = 30
n_tags = 5
print('input_dim: ', input_dim, '\noutput_dim: ', output_dim, '\ninput_length: ', input_length, '\nn_tags: ', n_tags)

def get_bilstm_lstm_model():
    model = Sequential()

    # Add Embedding layer
    model.add(Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length))

    # Add bidirectional LSTM
    model.add(Bidirectional(LSTM(units=output_dim, return_sequences=True, dropout=0.2, recurrent_dropout=0.2), merge_mode = 'concat'))

    # Add LSTM
    model.add(LSTM(units=output_dim, return_sequences=True, dropout=0.5, recurrent_dropout=0.5))

    # Add timeDistributed Layer
    model.add(TimeDistributed(Dense(n_tags, activation="relu")))

    #Optimiser 
    adam = k.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999)

    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy', f1_m, precision_m, recall_m])
    model.summary()
    
    return model

model = get_bilstm_lstm_model()
X_train, X_test, y_train, y_test = get_data()

print("[INFOR]: Shape of data train: ", X_train.shape)
# print("[INFOR]: Shape of data test: ", X_test.shape)
print("[INFOR]: Shape of label train: ", y_train.shape)
# print("[INFOR]: Shape of label test: ", y_test.shape)

# def train_model(X, y, model):
#     loss = list()
#     for i in range(25):
#         # fit model for one epoch on this sequence
#         hist = model.fit(X, y, batch_size=1000, verbose=1, epochs=1, validation_split=0.2)
#         loss.append(hist.history['loss'][0])
#     return loss
print(X_train[0])
hist =  model.fit(X_train, y_train, batch_size=10, epochs=10, validation_data=(X_test, y_test))

model.save('ner.h5')
# results = pd.DataFrame()
# model_bilstm_lstm = get_bilstm_lstm_model()
# plot_model(model_bilstm_lstm)
# results['with_add_lstm'] = train_model(train_tokens, np.array(train_tags), model_bilstm_lstm)
test = X_test[0]
test = np.expand_dims(test, axis = 0)
print(test.shape)

pred = model.predict(test)
print(pred)

# print(y_test[0])
# train_model(X_train, y_train, model)
# results = pd.DataFrame()
# model_bilstm_lstm = get_bilstm_lstm_model()

# results['with_add_lstm'] = train_model(X_train, np.array(y_train), model_bilstm_lstm)
# print(results)