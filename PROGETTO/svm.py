import os
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
from matplotlib import cm
from numpy import asarray, load
import numpy as np
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import pandas as pd

photos = load('D:\\Utenti\\PROGETTO_ML\\foto.npy')
labels = load('D:\\Utenti\\PROGETTO_ML\\labels.npy')
print(photos.shape, labels.shape)

xtrain, xtest, ytrain, ytest = train_test_split(photos, labels, test_size=0.2)
nsamples, nx, ny ,c= xtrain.shape
d2xtrain=xtrain.reshape((nsamples,nx*ny))
nsamples, nx, ny ,c= xtest.shape
d2xtest=xtest.reshape((nsamples,nx*ny))


model_params = {
    'svm': {
        'model': svm.SVC(gamma='auto'),
        'params' : {
            'C': [1,10,20],
            'kernel': ['linear','rbf']
        }
    },
}

scores = []

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=10)
    clf.fit(d2xtrain, ytrain)
    scores.append({
      'model': model_name,
       'best_score': clf.best_score_,
            'best_params': clf.best_params_
    })

from joblib import dump,load
#dump(clf, 'D:\\Utenti\\PROGETTO_ML\\modelliAddestrati\\svm.pkl')
#clf= load('D:\\Utenti\\PROGETTO_ML\\modelliAddestrati\\svm.pkl')

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df)
accuracy = clf.score(d2xtest, ytest)
print(accuracy)

y_pred = clf.predict(d2xtest)

print(classification_report(ytest, y_pred, target_names=[ 'jeans', 'shirts', 'trousers', 'watches']))
print(confusion_matrix(ytest, y_pred))
