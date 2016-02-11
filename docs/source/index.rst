.. YouTubeFacesDB documentation master file, created by
   sphinx-quickstart on Wed Feb 10 18:14:23 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

YouTubeFacesDB
==========================================

Python module allowing to load the YouTube Faces Database:

http://www.cs.tau.ac.il/~wolf/ytfaces/


Installation
-----------------

Apart from the usual python (2.7) + numpy dependencies, the module requires:

* **Pillow** ``pip install Pillow --user`` for image processing.
* **h5py** ``pip install h5py --user`` to manage the HDF5 files. ``libhdf5`` should also be installed through your package manager.

The module can then be installed locally with:

.. code-block:: bash

    python setup.py install --user

To build the documentation, you will need **Sphinx** ``pip install Sphinx --user``. You can then go into the ``docs/`` directory and build it with:

.. code-block:: bash

    make html

Generate a subset of YTFDB
----------------------------------

.. autofunction:: YouTubeFacesDB.generate_ytf_database

Loading a HDF5 file
------------------------

.. autoclass:: YouTubeFacesDB.YouTubeFacesDB
    :members:


