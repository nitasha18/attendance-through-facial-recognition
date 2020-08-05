# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 17:10:48 2018

@author: NITASHA GUPTA
"""

import cv2
import os
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
face_detector = cv2.CascadeClassifier('D:\OpenCV\opencv\sources\data\haarcascades_cuda\haarcascade_frontalface_default.xml')
#for each person, enter one numeric face id
face_id = input('\n Enter user id end press <return> ==>  ')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.2, 10)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        #Save the captured image into the dataSets folder
        cv2.imwrite("C:/Users/NITASHA GUPTA/.spyder-py3/FacialRecognitionAttendanceProject/DataSet/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imshow('image', img)
    
    k = cv2.waitKey(100) & 0xff
    
    if k== ord('q') or k==27:   #press 'q' key or ESC key
        break
    elif count >= 30: # Take 30 face sample and stop video
         break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()