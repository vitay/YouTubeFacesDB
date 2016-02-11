# YouTubeFacesDB

Python module allowing to load the YouTube Faces Database:

http://www.cs.tau.ac.il/~wolf/ytfaces/

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

## Usage

### Transforming the YouTube Faces Database into a HDF5 file

An example is provided in `examples/GenerateSubset.py`.

### Loading the HDF5 file for usage in Python
