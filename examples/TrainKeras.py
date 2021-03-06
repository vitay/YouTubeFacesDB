from __future__ import print_function, with_statement
from time import time

from YouTubeFacesDB import YouTubeFacesDB

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from keras.regularizers import l2, activity_l2
from keras.utils import np_utils


###############################################################################
# Load the data from disk
###############################################################################
tstart = time()

db = YouTubeFacesDB('ytfdb.h5', mean_removal=True, output_type='vector')
N = db.nb_samples
d = db.input_dim
C = db.nb_classes

print(N, 'images of size', d, 'loaded in', time()-tstart)

###############################################################################
# Split into a training set and a test set 
###############################################################################
db.split_dataset(validation_size=0.25)
X_train, y_train = db.get('train')
X_test, y_test = db.get('val')

###############################################################################
# Train a not very deep network
###############################################################################
print('Create the network...')
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
	model.fit(X_train, y_train,
          batch_size=100, nb_epoch=10,
          show_accuracy=True, verbose=2,
          validation_data=(X_test, y_test))
except (KeyboardInterrupt, ):
	pass

# Test on the validation set
score = model.evaluate(X_test, y_test,
                       show_accuracy=True, verbose=2)

print('Training finished.')
print('Test score:', score[0])
print('Test accuracy:', score[1])