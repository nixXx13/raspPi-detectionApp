import numpy as np

class Constants():

    # type np.ndarr
    # if None , first frame will be base
    BASE = None # np.load("base.npy") # None

    SOURCE = 1

    MOTION_THRESHOLD = 30
    OPENNING_KERNEL = np.ones((10,10),np.uint8)
    DILATION_KERNEL = np.ones((10,10),np.uint8)

    BOUNDING_BOX_THRESHOLD = 100*100

    MODEL_PATH = 'my_model9792.h5'
