# carica il dataset, reshape and save to a new file
from os import listdir
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from PIL import Image, ImageOps

# define location of dataset
folder = 'D:\\Utenti\\PROGETTO_ML\\immagini2\\'
photos, labels = list(), list()

# enumerate files in the directory
for file in listdir(folder):
	# determine class
	output = 0
	if file.startswith('jeans'):
		output = 1
	if file.startswith('shirts'):
		output = 2
	if file.startswith('trousers'):
		output = 3
	if file.startswith('watches'):
		output = 4
	# load image
	photo = load_img(folder + file, target_size=(60, 60))
	# converti in bianconero
	im = ImageOps.grayscale(photo)
	# convert PIL to numpy array
	photo = img_to_array(im)
	#normalizza valori
	photo=photo/255
	# store
	photos.append(photo)
	labels.append(output)
# convert to a numpy arrays
photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)
# save the reshaped photos
save('D:\\Utenti\\PROGETTO_ML\\foto.npy', photos)
save('D:\\Utenti\\PROGETTO_ML\\labels.npy', labels)

