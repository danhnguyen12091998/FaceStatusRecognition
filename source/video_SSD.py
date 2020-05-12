# import library
from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np
from imutils.video import VideoStream
import argparse
import time
import os

print ('[INFO] using opencv version %s]' % cv2.__version__)
# parameters for loading data and models
print("[INFO] loading face detection")
proto_path = 'SSD/deploy.prototxt.txt'
model_path = 'SSD/res10_300x300_ssd_iter_140000.caffemodel'
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5)
args = vars(ap.parse_args())
detector = cv2.dnn.readNetFromCaffe(proto_path, model_path)
print("[INFO] loading face emotion")
emotion_model_path = 'models/models_mini_XCEPTION.40-0.80.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)
# EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
#  "neutral"]
EMOTIONS = ["sad", "happy", "neutral"]
#loading video 
print("[INFO] loading video ... ")
video = cv2.VideoCapture('FaceEmotion_ID/video/demo_1.avi')
one_sec = 0.0
number_of_frames = 0
#process image
while True:
    start_time_of_frame = time.time()
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
            #print("[INFO] FPS: ", 1.0 / (time.time() - start_time))
            number_of_frames = number_of_frames + 1
            time_of_frame = time.time() - start_time_of_frame 
            one_sec = one_sec + time_of_frame
            if one_sec == 1.0 :
                print("[INFO] FPS: ", number_of_frames)
                number_of_frames = 0
                one_sec = 0.0
            elif one_sec > 1.0 :
                print("[INFO] FPS: ", number_of_frames)
                number_of_frames = 0
                one_sec = one_sec - 1.0
            else :
                continue
            # ensure the face width and height are sufficiently large
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()