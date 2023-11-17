from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from numpy import asarray, load
from sklearn.metrics import classification_report, confusion_matrix

photos = load('D:\\Utenti\\PROGETTO_ML\\foto.npy')
labels = load('D:\\Utenti\\PROGETTO_ML\\labels.npy')
print(photos.shape, labels.shape)

xtrain, xtest, ytrain, ytest = train_test_split(photos, labels, test_size=0.2, random_state=0)

nsamples, nx, ny ,c= xtrain.shape
d2xtrain=xtrain.reshape((nsamples,nx*ny))
nsamples, nx, ny ,c= xtest.shape
d2xtest=xtest.reshape((nsamples,nx*ny))



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
    clf.fit(d2xtrain, ytrain)
    scores.append({
      'model': model_name,
       'best_score': clf.best_score_,
            'best_params': clf.best_params_
    })

y_pred = clf.predict(d2xtest)

from joblib import dump,load
dump(clf, 'D:\\Utenti\\PROGETTO_ML\\modelliAddestrati\\naiveBayes.pkl')

print(classification_report(ytest, y_pred, target_names=[ 'jeans', 'shirts', 'trousers', 'watches']))
print(confusion_matrix(ytest, y_pred))