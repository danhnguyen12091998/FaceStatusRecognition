#! /usr/bin/env python3
# -*-coding: utf-8-*-

# USAGE
# python cut_face.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel
__author__ = 'Nghia Dap Troai'
import glob
import numpy as np
import cv2
import mxnet as mx
import pandas as pd
import random
import os
import argparse
import imutils

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Definded
def unique_name(folderName, suffix='jpg'):
    filename = '{0}_{1}.{2}'.format("Training7",random.randint(1,10**8),suffix)
    filepath = os.path.join(folderName,filename)
    if not os.path.exists(filepath):
        return filepath
    unique_name(folderName)


# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
print(args['model']) # args['model] lưu tên của model (args chứa tên của input đầu vào)

curdir = os.path.abspath(os.path.dirname(__file__)) # lấy vị trí thư mục hiện hành
print(curdir)
folderInput = os.path.join(curdir,'ImagesInput')        # Path  ImagesInput Folder
folderOutput = os.path.join(curdir, 'ImagesDetected')    # Path ImagesDetected Folder

print(folderInput)
print(folderOutput)
cv_img = []
for img in glob.glob(folderInput + "/*.jpg"):
    image = cv2.imread(img)
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))
    print("Value blob...........")
    print(blob.shape)
    print("[INFO] computing object detections...")
    net.setInput(blob)
    detections = net.forward()
    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is greater than the minimum confidence
        if confidence > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            #------------------------------------------------------
            length = abs(endY - startY - endX + startX)
            if (length % 2) == 0:
                startX = startX - int(length/2)
                endX = endX + int(length/2)
            else:
                startX = startX - int(length/2)
                endX = endX + int(length/2) + 1
            # ở trên đã tính toán các tọa độ phù hợp để có thể cắt ra khuông mặt:
            # Các tọa độ cần thiết đã được tính toán sẵn với các giá trị ở trong biến: (startX, endX), (startY, endY)
            # So với các hàm có sẵn thì không cần phải vẽ hình tròn
            # Ở đây cần thêm hàm để nhận diệu hết các inpit và lưu chúng vào thư mục
    #--------------------------------------------
    # Hàm để cắt ảnh và resize lại ảnh đ có kết quả mong muốn
    # output_Resize là giá trị của ảnh dữ liệu có kích thước 48x48 và đã được chuyển sang màu gray
    imagedetected = image[startY:endY, startX:endX]
    output_Resized = imutils.resize(imagedetected, 48, 48)
    print(output_Resized.shape)
    output_Resized =cv2.cvtColor(output_Resized, cv2.COLOR_BGR2GRAY)
    cv_img.append(output_Resized)

# Save all images output to ImagesDetected
for imge in cv_img:
    filepath = unique_name(folderOutput)
    print('^^ Write image to %s' % filepath)
    cv2.imwrite(filepath,imge)
