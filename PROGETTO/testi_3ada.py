import joblib
from numpy import load
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


X=load('D:\\Utenti\\PROGETTO_ML\\text\\x.npy',allow_pickle='true')
y=load('D:\\Utenti\\PROGETTO_ML\\text\\y.npy',allow_pickle='true')

print(X.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

DTC = DecisionTreeClassifier()

model_params = {
   'adaboost': {
       'model': AdaBoostClassifier(base_estimator= DTC),
       'params': {
           'base_estimator__criterion' : ['entropy','gini'],
           'base_estimator__splitter' : ['best','random'],
           'n_estimators' : [5,10,20,30,40,50]
       }
   },
}

scores = []

for model_name, mp in model_params.items():
  clf = GridSearchCV(mp['model'], mp['params'], cv=10, return_train_score=False)
  clf.fit(X_train, y_train)
  scores.append({
   'model': model_name,
   'best_score': clf.best_score_,
   'best_params': clf.best_params_
   })

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df)
print(clf.best_score_)
print(clf.best_params_)
from joblib import dump,load
#dump(clf, 'D:\\Utenti\\PROGETTO_ML\\modelliAddestrati\\testiADA2.pkl')
clf= load('D:\\Utenti\\PROGETTO_ML\\modelliAddestrati\\testiADA2.pkl')

y_pred = clf.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
