from feature_extract.tf_idf import extract_feature_for_train
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import pickle
X_train, y_train, X_test, y_test = extract_feature_for_train()

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(256,64),max_iter= 50000).fit(X_train, y_train)

predict = mlp.predict(X_test)
from sklearn.metrics import accuracy_score
print("Hieu qua mo hinh dat :", 100* accuracy_score(y_test, predict.tolist() ) )
from sklearn.metrics import f1_score

print("F1-score: ", f1_score(y_test, predict.tolist(), average='macro'))

from sklearn.metrics import confusion_matrix

print( confusion_matrix(y_test, predict.tolist() ))

plot_confusion_matrix(mlp, X_test, y_test)

plt.show()

model_path = 'MLP_model_nlu.pickle'

pickle.dump(mlp, open(model_path, 'wb'))