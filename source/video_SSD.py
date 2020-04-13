# import library
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import time
import os

# parameters for loading data and models
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
        help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
        help="path to Caffe pre-trained model")
ap.add_argument("-n", "--mini", required=True,
        help="path to mini xception model")
ap.add_argument("-v", "--video", required=True,
        help="path to video")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
        help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
print("[INFO] loading face detection")
detector = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
print("[INFO] loading face emotion")
emotion_classifier = load_model(args["mini"], compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]
#loading video 
print("[INFO] loading video ... ")
video = cv2.VideoCapture(args["video"])
#process image
while True:
    ret, frame = video.read()
    frame = imutils.resize(frame, width=800)
    (h, w) = frame.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300, 300)
    ,(104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]
        # filter out weak detections
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            ##roi = gray[startY:endY, startX:endX]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = gray[startY:startY + fH, startX:startX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            # ensure the face width and height are sufficiently large
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
fps.stop()
cv2.destroyAllWindows()
vs.stop()
