from utils import showImage, USBCamera, Models
from utilsFunctions import targetFunctions
from Constants import Constants
import numpy as np
import ImgOps
import cv2

def Main():

    src = USBCamera()
    model = Models.getModel()
    base = Constants.BASE

    iterNum = 1

    ret, frame = src.read()
    while ret:

        frame           = ImgOps.resize(frame)
        frame_grey      = ImgOps.grayscale(frame)
        frame_gauss     = ImgOps.gaussianBlur(frame_grey)

        if base is None:
            base = frame_gauss
            # np.save("base.npy", base)

        frame_diff      = ImgOps.subtract(frame_gauss,base)
        frame_thresh    = ImgOps.threshold(frame_diff)
        frame_open      = ImgOps.openning(frame_thresh)

        cv2.imshow("frame_diff", frame_diff)        if Constants.VERBOSE_UI else None

        contours = ImgOps.getContours(frame_open)

        for contour in contours:
            x, y, w, h = ImgOps.rectOfContour(contour)

            if w*h > Constants.BOUNDING_BOX_THRESHOLD and w*h != Constants.FULL_IMAGE_SIZE:

                framex = frame[y:y + h, x:x + w]
                framex = ImgOps.resize(framex, Constants.KERAS_INPUT_DIMS)
                cv2.imshow("framex", framex) if Constants.VERBOSE_UI else None
                framex = np.expand_dims(framex,axis=0)

                probs = model.predict(framex)
                targetFunctions.printClassification(probs)

        cv2.imshow("frame", frame)
        cv2.waitKey(1)

        iterNum +=1
        ret, frame = src.read()

    print("Releasing resources and closing opened windows")
    src.release()
    cv2.destroyAllWindows() if Constants.VERBOSE_UI else None


if __name__ == '__main__':
    Main()
