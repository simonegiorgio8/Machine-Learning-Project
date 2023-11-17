from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from numpy import asarray, load


X=load('D:\\Utenti\\PROGETTO_ML\\text\\x.npy',allow_pickle='true')
y=load('D:\\Utenti\\PROGETTO_ML\\text\\y.npy',allow_pickle='true')

print(X.shape)
print(y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model_params = {
    'svm': {
        'model': GaussianNB(),
        'params' : {
            'var_smoothing': [1e-9]
        }
    },
}

scores = []

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['params'], cv=10)
    clf.fit(X_train, y_train)
    scores.append({
      'model': model_name,
       'best_score': clf.best_score_,
            'best_params': clf.best_params_
    })

y_pred = clf.predict(X_test)

from joblib import dump,load
dump(clf, 'D:\\Utenti\\PROGETTO_ML\\modelliAddestrati\\naiveBayesTESTI.pkl')

print(classification_report(y_test, y_pred, target_names=[ 'ham', 'spam']))
print(confusion_matrix(y_test, y_pred))