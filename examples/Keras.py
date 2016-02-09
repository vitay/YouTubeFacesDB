from __future__ import print_function, with_statement
from time import time
import numpy as np
import h5py

from YouTubeFacesDB import generate_ytf_database, YouTubeFacesDB

db_filename = 'ytfdb_100_100_bw.h5'

# Create the database
generate_ytf_database(	
	directory='../data', 
	filename=db_filename,
    labels=['George_W_Bush', 'Bill_Clinton'],
    max_number=-1,
	size=(100, 100),
	color=True,
	rgb_first=True,
	cropped=True
)

# Load the data
tstart = time()
db = YouTubeFacesDB(db_filename)
X, y = db.get_whole_data()
N = db.nb_samples
d = db.input_dim

print(N, d)

print('Data loaded in', time()-tstart)

# Normalize inputs
mean_face = db.mean
X -= mean_face

