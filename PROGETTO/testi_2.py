import joblib
from numpy import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm
import pandas as pd

X=load('D:\\Utenti\\PROGETTO_ML\\text\\x.npy',allow_pickle='true')
y=load('D:\\Utenti\\PROGETTO_ML\\text\\y.npy',allow_pickle='true')

print(X.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)




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
 clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
 clf.fit(X_train, y_train)
 scores.append({
    'model': model_name,
    'best_score': clf.best_score_,
    'best_params': clf.best_params_
 })

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
print(df)

from joblib import dump,load
#dump(clf, 'D:\\Utenti\\PROGETTO_ML\\modelliAddestrati\\testiSvm.pkl')
clf= load('D:\\Utenti\\PROGETTO_ML\\modelliAddestrati\\testiSvm.pkl')
y_pred = clf.predict(X_test)


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
