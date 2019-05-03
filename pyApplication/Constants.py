import numpy as np

class Constants():
    MOTION_THRESHOLD = 20
    OPENNING_KERNEL = np.ones((10,10),np.uint8)
    DILATION_KERNEL = np.ones((10,10),np.uint8)