#import sklearn

#vectorizer = sklearn.feature_extraction.text.CountVectorizer();

#x= vectorizer.fit_[transform([doca,docb])]
#x.todense() #poichè matrici sparse di parole nei dizionari grandissimi, così mi restituisce un vettore per ogni doc, con i count


#vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(stop_words = 'english');
#x= vectorizer.fit_[transform([doca,docb])]
#x.todense() #questa volta non sono dei conteggi ma sono delle frequenze calcolate

from os import listdir

import joblib
import numpy as np
import nltk
import sklearn

nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from numpy import asarray, load
from numpy import save
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

#folder_ham = 'D:\\Utenti\\PROGETTO_ML\\testi2\\ham' #ham classe 0
#folder_spam = 'D:\\Utenti\\PROGETTO_ML\\testi2\\spam' #spam classe 1
#docs, labels = list(), list()

#doc = np.genfromtxt("D:\\Utenti\\PROGETTO_ML\\testi2\\ham\\0001.2000-01-17.beck.ham.txt", dtype='str')
#print(doc)
mail_data = load_files(r"D:\\Utenti\\PROGETTO_ML\\testi2")
X, y = mail_data.data, mail_data.target

mail_data = asarray(X)
mail_labels = asarray(y)


for d in range(0, len(mail_data)):
    document = nltk.re.sub(r'\\n',' ', str(mail_data[d])) #rimuovo \n
    document = nltk.re.sub(r'\W', ' ', document) #rimuovo caratteri speciali
    document = nltk.re.sub(r'\s+[a-zA-Z]\s+', ' ', document) #caratteri singoli
    document = nltk.re.sub(r'\^[a-zA-Z]\s+', ' ', document) #rimuovo singoli car dall'inizio
    document = nltk.re.sub(r'\s+', ' ', document, flags=nltk.re.I)#tolgo spazi multipli
    # Removing prefixed 'b'
    document = nltk.re.sub(r'^b\s+', '', document)
    # Converting to Lowercase
    document = document.lower()
    #lemmatizzazione: sostituisce con radice
    document = document.split()
    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    mail_data[d]=document


tfidfconverter = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7,stop_words=stopwords.words('english'))
X = tfidfconverter.fit_transform(mail_data).toarray()

save('D:\\Utenti\\PROGETTO_ML\\text\\x.npy', X)
save('D:\\Utenti\\PROGETTO_ML\\text\\y.npy', mail_labels)




