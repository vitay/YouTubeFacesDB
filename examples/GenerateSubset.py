from YouTubeFacesDB import generate_ytf_database

###############################################################################
# Create the dataset
###############################################################################
generate_ytf_database(  
    directory= '/scratch/vitay/Datasets/YouTubeFaces', # Location of the YTF dataset
    filename='ytfdb.h5', # Name of the HDF5 file to write to
    labels=10, # Number of labels to randomly select
    max_number=10000, # Maximum number of images to use
    size=(100, 100), # Size of the images
    color=False, # Black and white
    rgb_first=False, # Useless for BW images
    bw_first=True, # Final shape is (1, w, h)
    cropped=True # The original images are cropped to the faces
)