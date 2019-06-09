import numpy as np

class Constants():

    VERBOSE = True
    VERBOSE_UI = True

    # type np.ndarr
    # if None , first frame will be base
    BASE = None# np.load("base.npy") # None

    SOURCE = 1

    MOTION_THRESHOLD = 30
    OPENNING_KERNEL = np.ones((10,10),np.uint8)
    DILATION_KERNEL = np.ones((10,10),np.uint8)

    BOUNDING_BOX_THRESHOLD = 25*25

    CV_DIMS = (240,320,3)
    FULL_IMAGE_SIZE = CV_DIMS[0]*CV_DIMS[1]

    KERAS_INPUT_DIMS = (224,224,3)

