# Standard library
from __future__ import print_function, with_statement
from time import time
import re
import os
import random
import csv
# Dependencies
import numpy as np
import h5py
from PIL import Image


class YouTubeFacesDB(object):
    """
    Class allowing to interact with a HDF5 file containing a subset of the Youtube Faces dataset.
    """
    def __init__(self, filename):
        """
        Parameters:
        * `filename`: path to the HDF5 file containing the data.
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
        self.mean = np.array(self.f.get('mean'))

        # Size
        shape = self._X.shape
        self.nb_samples = shape[0]
        self.input_dim = shape[1:]

        # Indices
        self._indices = list(range(self.nb_samples))
        self._training_indices = self._indices
        self._validation_indices = []
        self-_test_indices = []

        # Labels
        labels = self.f.get('labels')
        self.labels = []
        for label in labels:
            self.labels.append(str(label[0]))
        self.nb_classes = len(self.labels)

        # Video indices
        self.video = self.f.get('video')

    def split_dataset(self, validation_size=0.2, test_size=0.0):
        """
        Split the dataset into a training set, a validation set and optionally a test set.

        Parameters:

        * `validation_size`: proportion of the data in the validation set (default: 0.2)
        * `test_size`: proportion of the data in the test set (default: 0.0) 

        The split is only internal to the object (the method returns nothing), as the actual data should be later read from disk. 

        To actually get the data, you will have to call either::

            X, y = db.get('all')
            X_train, y_train = db.get('train')
            X_val, y_cal = db.get('val')
            X_test, y_test = db.get('test')
        """
        pass

    def get(self, dset='all'):
        """
        Returns the whole dataset as a tuple (X, y) of numpy arrays.

        Parameters:

        * `dset`: string in ['train', 'val', 'test', 'all'] for the desired part of the dataset (default: 'all').

        """
        if not dset in ['train', 'val', 'test', 'all']:
            print("Error: the `dset` argument to get() must be in ['train', 'val', 'test', 'all']")
            return None, None
        if dset == 'all':
            return np.array(self._X), np.array(self._y)
        elif dset == 'train':
            return np.array(self._X), np.array(self._y)
        elif dset == 'val':
            return np.array(self._X), np.array(self._y)
        elif dset == 'test':
            return np.array(self._X), np.array(self._y)

    def get_batch(self, N, dset='train', shuffle=True):
        """
        Returns randomly a group of samples of the DB as a (idx, X, y) tuple.

        Parameters:

        * `N`: number of samples to retrieve.
        * `dset`: string in ['train', 'val', 'test', 'all'] for the desired part of the dataset (default: 'all').
        * `shuffle`: defines if the samples should be contiguous or randomly chosen (default: True)
        """      
        pass
        


