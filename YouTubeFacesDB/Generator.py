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

# Structure of the YFT directory
original_folder = '/frame_images_DB/'
aligned_folder = '/aligned_images_DB/'

def _get_labels(directory):
	"Retrieves the list of labels from the aligned directory"
	return sorted(os.listdir(directory + aligned_folder), key=lambda s: s.lower())

def _check_labels(labels, directory):
	"Compares the provided list of labels to ones which exist."
	orig = _get_labels(directory)
	for label in labels:
		if not label in orig:
			print('Error:', label, 'does not exist in the YouTube Faces database.')
			exit(0)


def _gather_images_info(directory, labels, max_images_per_person):
	"Iterates over all labels and gets the filenames and crop information"
	data = []
	for name in labels:
		# Each image is described in frame_images_DB/Aaron_Eckhart.labeled_faces.txt
		data_file = directory + original_folder + name + '.labeled_faces.txt'
		# Read the file
		data_person = []
		try:
			with open(data_file, 'r') as csvfile:
				for entry in csv.reader(csvfile, delimiter=','):
					img_name = entry[0].replace('\\', '/')
					center_w, center_h = int(entry[2]), int(entry[3])
					size_w, size_h = int(entry[4]), int(entry[5])
					data_person.append({
						'name': name,
						'filename': img_name,
						'center': (center_w, center_h),
						'size': (size_w, size_h)
						})
		except Exception as e:
			print('Error: could not read', data_file)
			print(e)
			return data
		# Possibly select a maximal number of them
		if max_images_per_person == -1: # everything
			data.extend(data_person)
		else:
			data.extend(random.sample(data_person, max_images_per_person))

	return data

def _create_db(directory, metadata, labels, filename, size, color, rgb_first, cropped):
	"Main method to fetch all images into the hdf5 DB."
	# Total number of images
	nb_images = len(metadata)
	# Final size of the image
	if color:
		final_size = (3, ) # channel is first
	else:
		final_size = ()
	final_size += size
	print('Final size of the images:', final_size)
	# Initialize the hdf5 DB
	f = h5py.File(filename, "w")
	dset_X = f.create_dataset("X", (nb_images,) + final_size, dtype='i')
	dset_Y = f.create_dataset("Y", (nb_images,), dtype='i')
	dset_video = f.create_dataset("video", (nb_images,), dtype='i')
	# Iterate over all images
	for idx in range(nb_images):
		# Retrieve the info
		description= metadata[idx] # description
		name = description['name'] # name of the person
		y = labels.index(name) # corresponding index between 0 and 1594
		filename = description['filename'] # complete filename
		video_idx = int(re.findall(r'/([\d]+)/', filename)[0]) # index of the video
		center_w, center_h = description['center'] # center of the face
		size_w, size_h = description['size'] # size of the face
		# Get the image
		img_file_path = directory + original_folder + filename
		img = Image.open(img_file_path)
		# Crop the image to the face
		if cropped:
			img = img.crop((center_w - size_w/2, center_h - size_h/2, center_w + size_w/2, center_h + size_h/2))
		# Resize the image
		img = img.resize(size)
		# Color
		if not color:
			img = img.convert('L')
		# Get the numpy array
		img_data = np.array(img)
		# Swap the axes (to have (3, w, h))
		if color and rgb_first:
			img_data = img_data.swapaxes(0, 2)
		# Push it to the HDF5 file
		dset_X[idx, ...] = img_data
		dset_Y[idx] = y
		dset_video[idx] = video_idx

def generate_ytf_database(
	directory, 
	filename, 
	size, 
	labels=None,
	max_number=-1, 
	max_images_per_person=-1, 
	color=True, 
	rgb_first=True, 
	cropped=True):
	"""
	Method to generate a subset of the YouTube Faces database in a HDF5 file.

	Arguments:

	* `directory`: director where the YouTube Face DB is located.
	* `filename`: path and name of the hdf5 file where the DB will be saved.
	* `size`: (width, height) size for the extracted images.
	* `labels`: number or list of labels which should be used (default: None, for all labels).
	* `max_number`: maximum number of images (default: -1, all of them).
	* `max_images_per_person`: maximum number of images which should be extracted per person (default: -1, all images)
	* `color`: if the color channels should be preserved (default: True) 
	* `rgb_first`: if True, the numpy arrays of colored images will have the shape (3, w, h), otherwise (w, h, 3) (default: True). Useful for Theano backends.
	* `cropped`: if the images should be cropped around the detected face (default: True)
	"""
	tstart = time()
	# Get the labels
	if labels==None:
		print('Retrieving all labels...')
		labels = _get_labels(directory)
	elif isinstance(labels, int):
		print('Generating',  labels, 'labels randomly...')
		nb_labels = labels
		orig = _get_labels(directory)
		labels = sorted(random.sample(orig, nb_labels), key=lambda s: s.lower())
		for label in labels:
			print('\t', label)
	else:
		print('Checking the labels...')
		_check_labels(labels, directory)

	# Retrieve the metadata on all images
	print('Gathering image locations...')
	metadata = _gather_images_info(directory, labels, max_images_per_person)
	nb_images = len(metadata)
	print('Found', nb_images, 'images for', len(labels), 'people.')

	# Reduce the number of images
	if max_number != -1:
		print('Reducing this number to', max_number)
		metadata = random.sample(metadata, max_number)

	# Get all the images, crop/resize them, and save them into a hdf5 file
	_create_db(directory, metadata, labels, filename, size, color, rgb_first, cropped)
	print('Done in', time()-tstart, 'seconds.')
