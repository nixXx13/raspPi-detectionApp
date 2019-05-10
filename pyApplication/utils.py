from Constants import Constants
from PIL import Image
import cv2
import time

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
        for i in range(5):
            self.read()


