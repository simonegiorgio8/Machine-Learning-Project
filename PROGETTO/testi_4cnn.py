import joblib
from numpy import load
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
from tensorflow.keras import layers, models

X=load('D:\\Utenti\\PROGETTO_ML\\text\\x.npy',allow_pickle='true')
y=load('D:\\Utenti\\PROGETTO_ML\\text\\y.npy',allow_pickle='true')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


ann = models.Sequential([
        layers.Dense(1500, activation='relu'),
        layers.Dense(150, activation='relu'),
        layers.Dense(2, activation='sigmoid')
   ])

ann.compile(optimizer='SGD',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=10)

y_pred = ann.predict(X_test)

y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))
print(confusion_matrix(y_test, y_pred_classes))