from __future__ import print_function, with_statement
from time import time
import numpy as np
import h5py

from Generator import create_ytf_database

db_filename = '/scratch/vitay/Datasets/YouTubeFaces/ytfdb_100_100_bw.h5'

# Create the database
create_ytf_database(	
	directory='/scratch/vitay/Datasets/YouTubeFaces', 
	filename=db_filename,
	size=(100, 100),
	color=False,
	rgb_first=True,
	cropped=True
)

# Load the data
tstart = time()
f = h5py.File(db_filename, "r")
X = np.array(f.get('X'))
y = np.array(f.get('Y'))
print('Data loaded in', time()-tstart)

# Normalize inputs
X = X.astype('float32')
X /= 255.
mean_face = np.mean(X, axis=0)
X -= mean_face
