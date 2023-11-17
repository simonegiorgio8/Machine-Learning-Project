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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.metrics import roc_curve,roc_auc_score

photos = load('D:\\Utenti\\PROGETTO_ML\\foto.npy')
labels = load('D:\\Utenti\\PROGETTO_ML\\labels.npy')

xtrain, xtest, ytrain, ytest = train_test_split(photos, labels, test_size=0.2)

#dig=2 #classe più numerosa shirts
#I_dig = np.where(ytrain == dig)[0]
#x_dig = xtrain[I_dig]
#y_dig = [2] * x_dig.shape[0]  # 2 rappresenta la classe inlayer
# print(y_dig)
#x_training = x_dig
#print(x_training.shape)


def dataset(dig): #prende dig che è la classe che vogliamo sia inlayer, e size di ogni altra classe outlyer

  #Crea dataset con normal score id
  I_dig = np.where(ytrain == dig)[0]
  x_dig = xtrain[I_dig]
  y_dig = [0] * x_dig.shape[0] #0 rappresenta la classe inlayer
  x_training = x_dig

  y_test= [0]*len(ytest)
  for i in range(len(ytest)):
      if (ytest[i]!=dig): y_test[i]=1

  return x_training,y_dig,asarray(y_test)


#plt.show()



inlier_digit = 2 #classe shirts più numerosa

x_training, y_training, y_test = dataset(inlier_digit)

#xtraining tutte le immagini
#ytraininig, sono le etichette: 0 per la classe inlayer, 1 per tutte le altre
#print(x_training.shape)
#print(y_training.shape)
#print(y_training)

# SIZE OF LAYERS
n = 3600
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
x_training = x_training.reshape(x_training.shape[0],3600)


exists = os.path.isfile("D:\\Utenti\\PROGETTO_ML\\anomalyDet\\anomaly.hist");
if exists:
    with open('D:\\Utenti\\PROGETTO_ML\\anomalyDet\\anomaly.hist','rb') as hist_file:
        history_dict = pickle.load(hist_file);
        autoencoder.load_weights('D:\\Utenti\\PROGETTO_ML\\anomalyDet\\anomalyW')
else:
    history = autoencoder.fit(x_training, x_training, epochs=ep, batch_size=batches);
    history_dict = history.history;
    with open('D:\\Utenti\\PROGETTO_ML\\anomalyDet\\anomaly.hist','wb') as hist_file:
        pickle.dump(history_dict, hist_file);
    autoencoder.save_weights('D:\\Utenti\\PROGETTO_ML\\anomalyDet\\anomalyW');

exists = os.path.isfile('D:\\Utenti\\PROGETTO_ML\\anomalyDet\\anomalyout.npy');
if exists:
    outlierness = np.load('D:\\Utenti\\PROGETTO_ML\\anomalyDet\\anomalyout.npy')
else:
    outlierness = np.zeros(xtest.shape[0])
    for i in range(xtest.shape[0]):
        outlierness[i] = autoencoder.evaluate(xtest[i].reshape((1, 3600)), xtest[i].reshape((1, 3600)),
                                              verbose=1)


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