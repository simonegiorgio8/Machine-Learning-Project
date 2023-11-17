import sys
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pylab as plt
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from numpy import asarray, load
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import train_test_split

#ricordo etichetta 1 SPAM

X=load('D:\\Utenti\\PROGETTO_ML\\text\\x.npy',allow_pickle='true')
y=load('D:\\Utenti\\PROGETTO_ML\\text\\y.npy',allow_pickle='true')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

dig=0
I_dig = np.where(y_train == dig)[0]
x_dig = X_train[I_dig]
y_dig = [0] * x_dig.shape[0]  # 0 rappresenta la classe inlayer
# print(y_dig)
x_training = x_dig
x_training = x_training.reshape(x_training.shape[0],1500)

# SIZE OF LAYERS
n = 1500
l2 = 128
l3 = 64

# MODEL'S STRUCTURE
input = keras.Input(shape=(n,))
x = keras.layers.Dense(l2, activation ='relu')(input)
x = keras.layers.Dense(l3,activation = 'relu')(x)
encoder = keras.Model(input, x)
x = keras.layers.Dense(l2, activation='relu')(x)
x = keras.layers.Dense(n, activation='sigmoid')(x)
autoencoder = keras.Model(input, x)

autoencoder.compile(optimizer='adam', loss='mse')

ep = 100
batches = 32


exists = os.path.isfile("D:\\Utenti\\PROGETTO_ML\\anomalyDet\\testi\\anomalyTESTI.hist");
if exists:
    with open('D:\\Utenti\\PROGETTO_ML\\anomalyDet\\testi\\anomalyTESTI.hist"','rb') as hist_file:
        history_dict = pickle.load(hist_file);
        autoencoder.load_weights('D:\\Utenti\\PROGETTO_ML\\anomalyDet\\testi\\anomalyW')
else:
    history = autoencoder.fit(x_training, x_training, epochs=ep, batch_size=batches,validation_data = (X_test, X_test));
    history_dict = history.history;
    with open('D:\\Utenti\\PROGETTO_ML\\anomalyDet\\testi\\anomalyTESTI.hist','wb') as hist_file:
        pickle.dump(history_dict, hist_file);
    autoencoder.save_weights('D:\\Utenti\\PROGETTO_ML\\anomalyDet\\testi\\anomalyW');

exists = os.path.isfile('D:\\Utenti\\PROGETTO_ML\\anomalyDet\\testi\\anomalyout.npy');
if exists:
    outlierness = np.load('D:\\Utenti\\PROGETTO_ML\\anomalyDet\\testi\\anomalyout.npy')
else:
    outlierness = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        outlierness[i] = autoencoder.evaluate(X_test[i].reshape((1, 1500)), X_test[i].reshape((1, 1500)),verbose=1)


print(outlierness)
plt.figure(1)
plt.plot(outlierness,'.')
plt.xlabel('test id')
plt.ylabel('outlierness')
plt.show()
fpr, tpr, thresholds = roc_curve(y_test,outlierness)
auc = roc_auc_score(y_test,outlierness)

plt.figure(2)
plt.plot(fpr,tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('AUC = '+str(auc))
plt.show()