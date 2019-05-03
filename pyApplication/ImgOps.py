from Constants import Constants
import traceback
from PIL import Image
import numpy as np
import imutils
import cv2

def debug(func):

    def printArgsInfo(*args,**kwargs):
        try:
            return func(*args,**kwargs)
        except Exception as e:
            print("="*10 + " Function '{}' failed ".format(func.__name__) + "="*10)
            print("="*5 +  " args given   " + "="*5)
            for arg in args:
                print(type(arg)) if type(arg) != np.ndarray else print((type(arg),arg.shape))
            print("=" * 5 + " kwargs given " + "=" * 5)
            for k in kwargs.keys():
                val = kwargs[k]
                print(k,type(val)) if type(val) != np.ndarray else print(k,type(val),val.shape)
            raise Exception(e)

    return printArgsInfo

@debug
def grayscale(ndArr):
    im = Image.fromarray(ndArr)
    im = im.convert('LA')
    return np.array(im)[:,:,0]  # TODO - find better grayscale - imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)?

@debug
def resize(ndArr , dim = (640,320)):
    return imutils.resize(ndArr,width=dim[0],height=dim[1])

@debug
def gaussianBlur(ndArr , kernel = (15,15)):
    return cv2.GaussianBlur(ndArr, (kernel[0], kernel[1]), 0)

@debug
def subtract(ndArr1,ndArr2):
    return cv2.subtract(ndArr2,ndArr1)

@debug
def threshold(ndArr):
    return cv2.threshold(ndArr,Constants.MOTION_THRESHOLD,255,cv2.THRESH_BINARY_INV)[1]

@debug
def openning(ndArr):
    # erosion followed by dilation
    return cv2.morphologyEx(ndArr, cv2.MORPH_OPEN, Constants.OPENNING_KERNEL,iterations=5)

@debug
def dilation(ndArr):
    return cv2.dilate(ndArr,Constants.DILATION_KERNEL,iterations=5)

@debug
def contours(ndArr):
        return cv2.findContours(ndArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

@debug
def bounding_rect(ndArr,origArr):
    ndArrCopy = ndArr.copy()
    cv2.rectangle(ndArrCopy, (0, 0), ndArrCopy.shape[::-1], (255, 255, 255), 15)
    contours, _ = cv2.findContours(ndArrCopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        ndArr = origArr
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            ndArr = cv2.rectangle(ndArr, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return ndArr , contours