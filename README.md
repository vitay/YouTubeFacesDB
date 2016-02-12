# YouTubeFacesDB

Python module allowing to load the YouTube Faces Database:

<http://www.cs.tau.ac.il/~wolf/ytfaces/>

**Description:** The data set contains 3,425 videos of 1,595 different people. All the videos were downloaded from YouTube. An average of 2.15 videos are available for each subject. The shortest clip duration is 48 frames, the longest clip is 6,070 frames, and the average length of a video clip is 181.3 frames. 

**For TUC users:** the DB is already downloaded on cortex at `/work/biblio/youtube Faces DB` (with the spaces). Copy it to your machine (in `/scratch`, as it is over 25GB) and uncompress it.

**Author:** Julien Vitay <julien.vitay@informatik.tu-chemnitz.de>

**License:** MIT

.. toctree::
   :maxdepth: 2

   api

## Installation

Apart from the usual python (2.7) + numpy dependencies, the module requires:

* **Pillow** `pip install Pillow --user` for image processing.
* **h5py** `pip install h5py --user` to manage the HDF5 files. `libhdf5` should also be installed through your package manager.

The module can then be installed locally with:

~~~bash
python setup.py install --user
~~~

To build the documentation, you will need Sphinx `pip install Sphinx --user`. You can then go into the `docs/` directory and build it with:

~~~bash
make html
~~~

You can then access `docs/build/html/index.html` with your browser.

## Tutorial

### Transforming the YouTube Faces Database into a HDF5 file

An example is provided in `examples/GenerateSubset.py`. It accesses the dataset located at `/scratch/vitay/Datasets/YouTubeFaces` (`directory`), selects 10 random labels from it (`labels`), fetches all corresponding images (`max_number`), crops them to contains only the face area (`cropped`), transform them to luminance-based (`color`), resizes them to (100, 100) (`size`), prepends a dummy dimension to obtain a final numpy array of shape (1, 100, 100) (`bw_first`) and dumps them to the HDF5 file `ytfdb.h5` (`filename`).

~~~python
from YouTubeFacesDB import generate_ytf_database
generate_ytf_database(  
    directory= '/scratch/vitay/Datasets/YouTubeFaces', # Location of the YTF dataset
    filename='ytfdb.h5', # Name of the HDF5 file to write to
    labels=10, # Number of labels to randomly select
    max_number=-1, # Maximum number of images to use
    size=(100, 100), # Size of the images
    color=False, # Black and white
    bw_first=True, # Final shape is (1, w, h)
    cropped=True # The original images are cropped to the faces
)
~~~

Check the doc of `generate_ytf_database` to see other arguments to this function.

**Beware:** if you try to generate all color images of all labels with a size (100, 100), the process will take over half an hour and the HDF5 file will be over 50GB, so do not save it in your home directory.

.. autofunction:: YouTubeFacesDB.generate_ytf_database

### Loading the HDF5 file for usage in Python

Once the HDF5 file has been generated, you can use it in a Python for learning. An example is provided in `examples/TrainKeras.py`, where a convolutional network written in Keras (`pip install Theano --user && pip install keras --user`) is trained on the data contained in `ytfdb.h5`. 

#### Loading the dataset into memory

To load the data, you need to create a `YouTubeFacesDB` object, pass him the path the HDF5 file and call the `get()` option:

~~~python
from YouTubeFacesDB import YouTubeFacesDB
db = YouTubeFacesDB('ytfdb.h5')
X, y = db.get()
~~~

`X` is a numpy array containing all input images. The first index correspond to the image number, the remaining ones to the shape of the numpy array representing each image. This information can also be retrieved through the attributes of the object:

~~~python
N = db.nb_samples # number of samples, e.g. 10000
d = db.input_dim # shape of the images, e.g. (1, 100, 100)
~~~

`y` is a numpy array containing the label index for each image (in vectorized form, see *categorical outputs*). You can access the number of labels, as well as the list of labels easily:

~~~python
C = db.nb_classes # Number of classes
labels = db.labels # List of strings for the labels
~~~

#### Transforming the data

**Mean removal** 

`X` contains for each pixel a floating value between 0. and 1. (the conversion between integers [0..255] and floats [0...1] was done during the generation process). However, neural networks typically work much better when the input data has a zero mean. Fortunately, the mean input (i.e. the mean face) was also saved during the generation process. You can remove it from the input using:

~~~python
mean_face = db.mean
X -= mean_face
~~~

You can also tell the `YouTubeFacesDB` object to remove systematically this mean from the inputs:

~~~python
db = YouTubeFacesDB('ytfdb.h5', mean_removal=True)
X, y = db.get()
~~~

This way, `X` has a zero mean over the first axis, without needing to explicitly compute it. This is particularly useful when generating minibatches.

**Categorical outputs**

The outputs labels are originally integers between 0 and `db.nb_classes` - 1. To train neural networks, it often required to represent the output as binary arrays of length `db.nb_classes`. where only one element is 1 and the rest 0. For example, the third class among 10 would be represented by `0000000100`.This is the default representation returned by the `YouTubeFacesDB` object.

If you prefer to get the labels as integers in `y`, you can specify it in the constructor:

~~~python
db = YouTubeFacesDB('ytfdb.h5', output_type='integer')
~~~

The default value of `output_type` is `vector`.


#### Splitting the data into training, validation and test sets

`db.get()` returns by default the whole data. If you want to split this data into training, validation and test sets, you can call the method `split_dataset()`:

~~~python
db.split_dataset(validation_size=0.2, test_size=0.1)
~~~

In this example, the validation set will contain 20% of the samples and the test set 10%. The rest stays in the training set. The samples are randomly chosen in the data. To retrieve the corresponding data, provide an argument to `get()`:

~~~python
db.split_dataset(validation_size=0.2, test_size=0.1)
X_train, y_train = db.get('train')
X_val, y_val = db.get('val')
X_test, y_test = db.get('test')
~~~

By default, the validation set has 20% of the data and the test set 0%.

#### Generating minibatches

Loading the whole dataset in memory with `get()` defeats the purpose of storing a large-scale dataset in a HDF5 file. In practice, it is recommended to load only minibatches (of let's say 1000 samples) one at a time, process them, and 
