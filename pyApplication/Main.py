from ImgOps import resize, grayscale, gaussianBlur, subtract, threshold, openning, bounding_rect
from tensorflow.python.keras.models import load_model
from Constants import Constants
from utils import showImage, Source
import numpy as np
import scipy.misc
import cv2

def Main():

    src = Source()
    model = load_model(Constants.MODEL_PATH)
    base = Constants.BASE

    iterNum = 1

    ret, frame = src.read()
    while ret:

        frame           = resize(frame)
        frame_grey      = grayscale(frame)
        frame_gauss     = gaussianBlur(frame_grey)

        if base is None:
            base = frame_gauss

        frame_diff      = subtract(frame_gauss, base)
        frame_thresh    = threshold(frame_diff)
        frame_open      = openning(frame_thresh)

        # TODO - separate to multiple funcs
        bound_box, contours = bounding_rect(frame_open, frame)

        for contour in contours:
            # TODO - refactor
            x, y, w, h = cv2.boundingRect(contour)
            if w*h > Constants.BOUNDING_BOX_THRESHOLD:
                # classify bounding box
                # framex = frame[y:y + h, x:x + w]
                # framex = scipy.misc.imresize(framex, (224,224,3))

                # classify the whole image
                framex = scipy.misc.imresize(frame, (224,224,3))
                framex = np.expand_dims(framex,axis=0)
                probs = model.predict(framex)

                if probs[0][0]>probs[0][1]:
                    print("DOG!", probs[0])
                else:
                    print("HUMAN!", probs[0])

        cv2.imshow("frame_open", bound_box)
        cv2.waitKey(1)

        iterNum +=1
        ret, frame = src.read()

    src.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    Main()
