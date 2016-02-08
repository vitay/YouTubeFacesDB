import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "YouTubeFacesDB",
    version = "0.0.1",
    author = "Julien Vitay",
    author_email = "julien.vitay@gmail.com",
    description = ("Python scripts to load the YouTube Faces Database."),
    license = "MIT",
    keywords = "youtube faces database",
    url = "https://ai.informatik.tu-chemnitz.de/gogs/vitay/YouTubeFacesDB",
    packages=['YouTubeFacesDB'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
    ],
)