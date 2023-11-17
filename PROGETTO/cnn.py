import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray, load

photos = load('D:\\Utenti\\PROGETTO_ML\\foto.npy')
labels = load('D:\\Utenti\\PROGETTO_ML\\labels.npy')
labels=labels-1
print(labels)
print(photos.shape, labels.shape)

xtrain, xtest, ytrain, ytest = train_test_split(photos, labels, test_size=0.2)
nsamples, nx, ny ,c= xtrain.shape
d2xtrain=xtrain.reshape((nsamples,nx*ny))
nsamples, nx, ny ,c= xtest.shape
d2xtest=xtest.reshape((nsamples,nx*ny))
#print(xtrain.shape,ytrain.shape)
#print(xtest.shape,ytest.shape)
#photo = img_to_array(photos[6000])
#plt.imshow(photo)
#plt.show()
#nsamples, nx, ny ,c= photos.shape
#d2xtrain=photos.reshape((nsamples,nx*ny))


#ann = models.Sequential([
#        layers.Flatten(input_shape=(60,60,1)),
#        layers.Dense(360, activation='relu'),
#        layers.Dense(400, activation='relu'),
#        layers.Dense(60, activation='relu'),
#        layers.Dense(4, activation='softmax')
#   ])
#
#ann.compile(optimizer='SGD',
#             loss='sparse_categorical_crossentropy',
#             metrics=['accuracy'])

#history= ann.fit(xtrain, ytrain, epochs=5)#, validation_split=0.2)

#y_pred = ann.predict(xtest)
#y_pred_classes = [np.argmax(element) for element in y_pred]

#print("Classification Report: \n", classification_report(ytest, y_pred_classes))
#print(confusion_matrix(ytest, y_pred_classes))






cnn = models.Sequential([
   layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(60, 60, 1)),
   layers.MaxPooling2D((2, 2)),

   layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
   layers.MaxPooling2D((2, 2)),

   layers.Flatten(),
   layers.Dense(64, activation='relu'),
   layers.Dense(4, activation='softmax')
])

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(xtrain, ytrain, epochs=10)

y_pred = cnn.predict(xtest)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(ytest, y_pred_classes))
print(confusion_matrix(ytest, y_pred_classes))

cnn.save('D:\\Utenti\\PROGETTO_ML\\modelliAddestrati\\cnn.h5')