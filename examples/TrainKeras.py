from __future__ import print_function, with_statement
from time import time

from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from keras.regularizers import l2, activity_l2
from keras.utils import np_utils

from YouTubeFacesDB import YouTubeFacesDB

###############################################################################
# Load the data from disk
###############################################################################
tstart = time()

db = YouTubeFacesDB('ytfdb.h5')
X, y = db.get_whole_data()
N = db.nb_samples
d = db.input_dim
C = db.nb_classes
mean_face = db.mean

print(N, 'images of size', d, 'loaded in', time()-tstart)

# Normalize inputs
X -= mean_face

###############################################################################
# Split into a training set and a test set 
###############################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

###############################################################################
# Train a not very deep network
###############################################################################
print('Create the network...')
# Convert class vectors to binary class matrices (e.g. class 3 -> 0000...00000100)
Y_train = np_utils.to_categorical(y_train, C)
Y_test = np_utils.to_categorical(y_test, C)

# Create the model
model = Sequential()

# Convolutional input layer with maxpooling and dropout
model.add(Convolution2D(16, 6, 6, border_mode='valid', input_shape=d))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

# Fully connected with ReLU and dropout
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
print('Start training...')
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

print('Training finished.')
print('Test score:', score[0])
print('Test accuracy:', score[1])