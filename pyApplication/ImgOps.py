from utilsFunctions import inputOutputValidation
from Constants import Constants
from PIL import Image
import numpy as np
import scipy.misc
import cv2

@inputOutputValidation
def grayscale(ndArr):
    im = Image.fromarray(ndArr)
    im = im.convert('LA')
    return np.array(im)[:,:,0]  # TODO - find better grayscale - imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)?

def resize(ndArr , dim = Constants.CV_DIMS):
    return scipy.misc.imresize(ndArr, dim)

def gaussianBlur(ndArr , kernel = (15,15)):
    return cv2.GaussianBlur(ndArr, (kernel[0], kernel[1]), 0)

@inputOutputValidation
def subtract(ndArr1,ndArr2):
    return cv2.subtract(ndArr2,ndArr1)

@inputOutputValidation
def threshold(ndArr):
    return cv2.threshold(ndArr,Constants.MOTION_THRESHOLD,255,cv2.THRESH_BINARY_INV)[1]

def openning(ndArr):
    # erosion followed by dilation
    return cv2.morphologyEx(ndArr, cv2.MORPH_OPEN, Constants.OPENNING_KERNEL,iterations=5)

def dilation(ndArr):
    return cv2.dilate(ndArr,Constants.DILATION_KERNEL,iterations=5)

def getContours(ndArr):
    ndArrCopy = ndArr.copy()
    cv2.rectangle(ndArrCopy, (0, 0), ndArrCopy.shape[::-1], (255, 255, 255), 15)
    cntrs, _ = cv2.findContours(ndArrCopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cntrs

def rectOfContour(contour):
    return cv2.boundingRect(contour)

def bounding_rect(contours,origArr):
    ndArr = origArr
    if len(contours) > 1:
        for contour in contours:
            x, y, w, h = rectOfContour(contour)
            ndArr = cv2.rectangle(ndArr, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return ndArr
