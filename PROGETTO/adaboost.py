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

#ada_clf = AdaBoostClassifier(
#DecisionTreeClassifier(max_depth=1), n_estimators=50,
#algorithm="SAMME.R", learning_rate=0.5)
#ada_clf.fit(d2xtrain, ytrain)
#print(ada_clf.score())

DTC = DecisionTreeClassifier()

model_params = {
    'adaboost': {
        'model': AdaBoostClassifier(base_estimator= DTC),
        'params': {
            'base_estimator__criterion' : ['entropy',  'gini'],
            'base_estimator__splitter' : ['best', 'random'],
            'n_estimators' : [5,10,20,30,40,50]
        }
    },
}

scores = []

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=10, return_train_score=False)
    clf.fit(d2xtrain, ytrain)
    scores.append({
      'model': model_name,
      'best_score': clf.best_score_,
      'best_params': clf.best_params_
    })

#file1 = open("D:\\Utenti\\PROGETTO_ML\\model_selection_ADABOOST.txt","w")
#file1.write(scores)    ARGUMENT MUST BE STR, NOT LIST ERROR
#file1.close()



from joblib import dump,load
#dump(clf, 'D:\\Utenti\\PROGETTO_ML\\modelliAddestrati\\adaboost2.pkl')
clf= load('D:\\Utenti\\PROGETTO_ML\\modelliAddestrati\\adaboost2.pkl')
print(clf.best_score_)
df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df)
#df.to_pickle("D:\\Utenti\\PROGETTO_ML\\scores_adaboost.pkl")
#df.to_csv("D:\\Utenti\\PROGETTO_ML\\scores_adaboost.csv")
accuracy = clf.score(d2xtest, ytest)
print(accuracy)

y_pred = clf.predict(d2xtest)

print(classification_report(ytest, y_pred, target_names=[ 'jeans', 'shirts', 'trousers', 'watches']))
print(confusion_matrix(ytest, y_pred))

