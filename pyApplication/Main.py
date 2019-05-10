from tensorflow.python.keras.models import load_model
from utilsFunctions import targetFunctions
from utils import showImage, USBCamera
from Constants import Constants
import numpy as np
import ImgOps
import cv2

def Main():

    src = USBCamera()
    model = load_model(Constants.MODEL_PATH)
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

        cv2.imshow("frame_diff", frame_diff)
        cv2.imshow("frame_thresh", frame_thresh)
        cv2.imshow("frame_open", frame_open)

        contours = ImgOps.getContours(frame_open)

        for contour in contours:
            x, y, w, h = ImgOps.rectOfContour(contour)

            if w*h > Constants.BOUNDING_BOX_THRESHOLD and w*h != Constants.FULL_IMAGE_SIZE:

                framex = frame[y:y + h, x:x + w]
                framex = ImgOps.resize(framex, Constants.KERAS_INPUT_DIMS)
                cv2.imshow("framex", framex)
                framex = np.expand_dims(framex,axis=0)

                probs = model.predict(framex)
                targetFunctions.printClassification(probs)

        cv2.imshow("frame", frame)
        cv2.waitKey(1)

        iterNum +=1
        ret, frame = src.read()

    src.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    Main()
