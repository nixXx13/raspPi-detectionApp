import tensorflow.keras as keras
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import numpy as np
import imutils
import scipy.misc
from PIL import Image , ImageChops
from Constants import Constants as C
import cv2

print (cv2.__version__)
# print (tf.__version__)

filename="vid_ex0.mp4"
source=1
num = 0 # implement with a decorator

def saveImage(ndArr):
    im = Image.fromarray(ndArr)
    im.save("img{}.jpeg".format(num))
    num+=1

def showImage(ndArr):
    im = Image.fromarray(ndArr)
    im.show()

def grayscale(ndArr):
    im = Image.fromarray(ndArr)
    im = im.convert('LA')
    return np.array(im)[:,:,0]  # TODO - find better grayscale - imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)?

def resize(ndArr , dim = (640,320)):
    return imutils.resize(ndArr,width=dim[0],height=dim[1])

def gaussianBlur(ndArr , kernel = (15,15)):
    return cv2.GaussianBlur(ndArr, (kernel[0], kernel[1]), 0)

def subtract(ndArr1,ndArr2):
    return cv2.subtract(ndArr2,ndArr1)

def threshold(ndArr):
    return cv2.threshold(ndArr,C.MOTION_THRESHOLD,255,cv2.THRESH_BINARY_INV)[1]

def openning(ndArr):
    # erosion followed by dilation
    return cv2.morphologyEx(ndArr, cv2.MORPH_OPEN, C.OPENNING_KERNEL,iterations=5)

def dilation(ndArr):
    return cv2.dilate(ndArr,C.DILATION_KERNEL,iterations=5)

def contours(ndArr):
        return cv2.findContours(ndArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

def bounding_rect(ndArr,origArr):
    ndArrCopy = ndArr.copy()
    cv2.rectangle(ndArrCopy, (0, 0), ndArrCopy.shape[::-1], (255, 255, 255), 15)
    contours, _ = cv2.findContours(ndArrCopy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 1:
        cv2.drawContours(ndArrCopy, contours, 1, (125, 125, 125), 5)
        x, y, w, h = cv2.boundingRect(contours[1])
        ndArr = cv2.rectangle(origArr, (x, y), (x + w, y + h), (0, 255, 0), 3)
    return ndArr

cap = cv2.VideoCapture(source)

base = np.load("base.npy") # None
iter = 0
ret, frame = cap.read()
model = load_model('my_model9792.h5')

while(ret ==True):
    # if iter%90 == 0 :
    if True:

        # framex =  frame.reshape((1,)+frame.shape)
        # framex = np.resize(frame,(224,224,3))

        framex = scipy.misc.imresize(frame, (224,224,3))
        framex = np.expand_dims(framex,axis=0)
        probs = model.predict(framex)
        if(probs[0][0]>probs[0][1]):
            print("DOG!", probs[0])
        else:
            print("HUMAN!", probs[0])

        # frame           = resize(frame)
        # frame           = grayscale(frame)
        # frame_gauss     = gaussianBlur(frame)
        #
        # if base is None:
        #     base = frame_gauss
        #     # np.save("base",base)
        #
        # frame_diff      = subtract(frame_gauss, base)
        # print(np.max(frame_diff),np.min(frame_diff))
        # frame_thresh    = threshold(frame_diff)
        # frame_open      = openning(frame_thresh)
        #
        # b_box = bounding_rect(frame_open, frame)
        #
        # # showImage(b_box)
        cv2.imshow("frame_open", frame)
        cv2.waitKey(1)

    iter +=1
    # print("frame number {}".format(iter))
    ret, frame = cap.read()

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()