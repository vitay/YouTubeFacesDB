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
        self.X = self.f.get('X')
        self.y = self.f.get('Y')

        # Mean input
        self.mean = np.array(self.f.get('mean'))

        # Size
        shape = self.X.shape
        self.nb_samples = shape[0]
        self.input_dim = shape[1:]

        # Labels
        labels = self.f.get('labels')
        self.labels = []
        for label in labels:
            self.labels.append(str(label[0]))
        self.nb_classes = len(self.labels)

        # Video indices
        self.video = self.f.get('video')

    def get_whole_data(self):
        "Return the whole data as a tuple (X, y) of numpy arrays."
        return np.array(self.X), np.array(self.y)