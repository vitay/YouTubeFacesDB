# Standard library
from __future__ import print_function, with_statement
from time import time
import re
import os
import copy
import random
import csv
# Dependencies
import numpy as np
import h5py
from PIL import Image


def to_categorical(y, nb_classes=None):
    """
    Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.

    Taken from Keras.
    """
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y

class YouTubeFacesDB(object):
    """
    Class allowing to interact with a HDF5 file containing a subset of the Youtube Faces dataset.
    """
    def __init__(self, filename, mean_removal=False, output_type='vector'):
        """
        Parameters:
        
        * `filename`: path to the HDF5 file containing the data.
        * `mean_removal`: defines if the mean image should be substracted from each image.
        * `output_type`: ['integer', 'vector'] defines the output for each sample. 'integer' will return the index of the class (e.g. 3), while vector will return a vector ith nb_classes components, all zero but one (e.g. 000...00100). Default: vector. 
        """
        # Open the file
        self.filename = filename
        try:
            self.f = h5py.File(self.filename, "r")
        except Exception:
            print('Error:', self.filename, 'does not exist.')

        # Data
        self._X = self.f.get('X')
        self._y = self.f.get('Y')

        # Mean input
        self.mean_removal = mean_removal
        self.mean = np.array(self.f.get('mean'))

        # Size
        shape = self._X.shape
        self.nb_samples = shape[0]
        self.input_dim = shape[1:]

        # Indices
        self._indices = list(range(self.nb_samples))
        self._training_indices = self._indices
        self._validation_indices = []
        self._test_indices = []
        self.nb_train = self.nb_samples
        self.nb_val = 0
        self.nb_test = 0

        # Labels
        labels = self.f.get('labels')
        self.labels = []
        for label in labels:
            self.labels.append(str(label[0]))
        self.nb_classes = len(self.labels)
        if not output_type in ['integer', 'vector']:
            print("Error: output_type must be in ['integer', 'vector']")
            output_type = 'vector'
        self.output_type = output_type

        # Video indices
        self.video = self.f.get('video')

    def split_dataset(self, validation_size=0.2, test_size=0.0):
        """
        Split the dataset into a training set, a validation set and optionally a test set.

        Parameters:

        * `validation_size`: proportion of the data in the validation set (default: 0.2)
        * `test_size`: proportion of the data in the test set (default: 0.0) 

        The split is only internal to the object (the method returns nothing), as the actual data should be later read from disk. 

        This method sets the following attributes:

        * `self.nb_train`: number of samples in the training set.
        * `self.nb_val`: number of samples in the validation set.
        * `self.nb_test`: number of samples in the test set.

        To actually get the data, you will have to call either::

            X, y = db.get('all')
            X_train, y_train = db.get('train')
            X_val, y_cal = db.get('val')
            X_test, y_test = db.get('test')
        """
        # Number of examples
        self.nb_val = int(self.nb_samples*validation_size)
        self.nb_test = int(self.nb_samples*test_size)
        self.nb_train = self.nb_samples - self.nb_val - self.nb_test
        # Compute the indices
        indices = copy.deepcopy(self._indices)
        random.shuffle(indices)
        self._validation_indices = sorted(indices[:self.nb_val])
        if self.nb_test != 0:
            self._test_indices = sorted(indices[self.nb_val:self.nb_val+self.nb_test])
        else:
            self._test_indices = []
        self._training_indices = sorted(indices[self.nb_val+self.nb_test:])
        print('Training:', self.nb_train, '; Validation:', self.nb_val, '; Test:', self.nb_test, '; Total:', self.nb_samples)

    def get(self, dset='all'):
        """
        Returns the whole dataset as a tuple (X, y) of numpy arrays.

        Parameters:

        * `dset`: string in ['train', 'val', 'test', 'all'] for the desired part of the dataset (default: 'all').

        """
        if dset == 'all':
            X = np.array(self._X)
            y = np.array(self._y, dtype='int32')
        elif dset == 'train':
            X = np.array(self._X[self._training_indices, ...])
            y = np.array(self._y[self._training_indices, ...], dtype='int32')
        elif dset == 'val':
            X = np.array(self._X[self._validation_indices, ...])
            y = np.array(self._y[self._validation_indices, ...], dtype='int32')
        elif dset == 'test':
            X = np.array(self._X[self._test_indices, ...])
            y = np.array(self._y[self._test_indices, ...], dtype='int32')
        else:
            print("Error: the `dset` argument to get() must be in ['train', 'val', 'test', 'all']")
            X = np.array([[]])
            y = np.array([], dtype='int32')

        return self._transform_data(X, y)

    def _transform_data(self, X, y):
        "Applies transformations to the data (mean_removal, output type..."
        # Mean removal
        if self.mean_removal:
            X -= self.mean

        # Categorical outputs
        if self.output_type == 'vector':
            y = to_categorical(y, self.nb_classes)

        return X, y

    def generate_batches(self, batch_size, dset='all', rest=True):
        """
        Returns a minibatch of random samples of the DB as a (X, y) tuple every time it is called, until the dataset is fully seen.

        Parameters:

        * `batch_size`: number of samples per minibatch.
        * `dset`: string in ['train', 'val', 'test', 'all'] for the desired part of the dataset (default: 'all').
        * `rest`: defines if the remaining samples after the last full minibatch should be sent anyway (default: True)
        """     
        # Access the dataset indices 
        if dset=='train':
            indices = copy.deepcopy(self._training_indices)
            N = self.nb_train
        elif dset=='val':
            indices = copy.deepcopy(self._validation_indices)
            N = self.nb_val
        elif dset=='test':
            indices = copy.deepcopy(self._test_indices)
            N = self.nb_test
        elif dset=='all':
            indices = copy.deepcopy(self._indices)
            N = self.nb_samples
        else:
            print("Error: the `dset` argument to get_batch() must be in ['train', 'val', 'test', 'all']")
            return

        # Compute the number of minibatches
        nb_batches = int(N/batch_size)
        rest_batches = N - nb_batches*batch_size # what to do with the rest?

        # Shuffle the training set
        random.shuffle(indices)

        # Iterate over the minibatches
        for b in range(nb_batches):
            samples = sorted(indices[b*batch_size:(b+1)*batch_size])
            X = np.array(self._X[samples, ...])
            y = np.array(self._y[samples, ...], dtype='int32')
            X, y = self._transform_data(X, y)
            yield X, y

        # Throw the rest. May be inefficient.
        if rest_batches != 0 and rest:
            samples = sorted(indices[nb_batches*batch_size:])
            X = np.array(self._X[samples, ...])
            y = np.array(self._y[samples, ...], dtype='int32')
            X, y = self._transform_data(X, y)
            yield X, y



        


