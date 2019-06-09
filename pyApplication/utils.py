from tensorflow.keras.layers import Flatten,Dense,Conv2D,MaxPooling2D,Dropout,BatchNormalization,Convolution2D
from tensorflow.keras.models import Sequential
from Constants import Constants
from PIL import Image
import cv2

def showImage(ndArr):
    im = Image.fromarray(ndArr)
    im.show()

class Source:

    def __init__(self, source=Constants.SOURCE):
        print("Initializing video source from '{}'".format(source))
        self.source = cv2.VideoCapture(source)

    def read(self):
        return self.source.read()

    def release(self):
        self.source.release()


class USBCamera(Source):

    def __init__(self):
        super().__init__(source=Constants.SOURCE)

        # first frames of usb camera are dark and out of focus
        for i in range(15):
            self.read()

class Models():

    @staticmethod
    def getModel():
        filterCoef = 8
        denseCoef = 8
        model = Sequential([
            BatchNormalization(input_shape=(224, 224, 3)),
            Conv2D(filterCoef * 8, 3, activation='relu'),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            Conv2D(filterCoef * 16, 3, activation='relu'),
            MaxPooling2D((2, 2)),
            BatchNormalization(),
            Flatten(),
            Dense(denseCoef * 4, activation='relu'),
            BatchNormalization(),
            Dense(2, activation='softmax')
        ])
        model.load_weights("modelweights9792.h5")
        model.summary() if Constants.VERBOSE else None
        return model




