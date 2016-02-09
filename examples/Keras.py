from __future__ import print_function, with_statement
from time import time
import numpy as np
import h5py

from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from keras.regularizers import l2, activity_l2
from keras.utils import np_utils

from YouTubeFacesDB import generate_ytf_database, YouTubeFacesDB

db_filename = 'ytfdb_100_100_bw.h5'

###############################################################################
# Create the dataset
###############################################################################
generate_ytf_database(	
	directory='/scratch/vitay/Datasets/YouTubeFaces', 
	filename=db_filename,
    labels=30, #['George_W_Bush', 'Bill_Clinton'],
    max_number=-1, #1000,
	size=(100, 100),
	color=False,
	rgb_first=False,
	bw_first=True,
	cropped=True
)

###############################################################################
# Load the data from disk
###############################################################################
tstart = time()

db = YouTubeFacesDB(db_filename)
X, y = db.get_whole_data()
N = db.nb_samples
d = db.input_dim
C = db.nb_classes

print(N, 'images of size', d)

print('Data loaded in', time()-tstart)

# Normalize inputs
mean_face = db.mean
X -= mean_face

###############################################################################
# Split into a training set and a test set 
###############################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

###############################################################################
# Train a not very deep network
###############################################################################
# Convert class vectors to binary class matrices (e.g. class 3 -> 0000...00000100)
Y_train = np_utils.to_categorical(y_train, C)
Y_test = np_utils.to_categorical(y_test, C)

# Create the model
model = Sequential()

# Convolutional input layer ith maxpooling and dropout
model.add(Convolution2D(16, 6, 6, border_mode='valid', input_shape=d))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

# Fully connected ith ReLU and dropout
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Softmax output layer
model.add(Dense(C))
model.add(Activation('softmax'))

# Learning rule
optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# Training
try:
	model.fit(X_train, Y_train,
          batch_size=100, nb_epoch=10,
          show_accuracy=True, verbose=2,
          validation_data=(X_test, Y_test))
except (KeyboardInterrupt, ):
	pass

# Test on the validation set
score = model.evaluate(X_test, Y_test,
                       show_accuracy=True, verbose=2)

print('Test score:', score[0])
print('Test accuracy:', score[1])