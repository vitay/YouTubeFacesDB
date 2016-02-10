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
mean_face = db.mean

print(N, 'images of size', d, 'loaded in', time()-tstart)

###############################################################################
# Split into a training set and a test set 
###############################################################################
db.split_dataset(validation_size=0.25)
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
	nb_epochs = 10
	batch_size = 100
	for epoch in range(nb_epochs):
		tstart = time()
		# Training
		batch_generator = db.generate_batches(batch_size, dset='train')
		nb_batches = 0; training_loss = 0.0; training_accuracy = 0.0
		for X, y in batch_generator:
			loss, accuracy = model.train_on_batch(X, y, accuracy=True)
			training_loss += loss
			training_accuracy += accuracy
			nb_batches += 1
		# Validation
		score = model.evaluate(X_test, y_test,
                       show_accuracy=True, verbose=0)
		# Verbose
		print('Epoch', epoch+1, '/', nb_epochs)
		print('\tTraining loss:', training_loss/float(nb_batches), 'accuracy:', training_accuracy/float(nb_batches))
		print('\tValidation loss:', score[0], 'accuracy:', score[1])
		print('\tTook', time()-tstart)

except (KeyboardInterrupt, ):
	pass

# Validation
print('Training finished.')
score = model.evaluate(X_test, y_test,
               show_accuracy=True, verbose=2)
print('Test loss:', score[0], 'accuracy:', score[1])