# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 17:55:02 2018

@author: NITASHA GUPTA
"""

import cv2
import numpy as np
import os 
import datetime

'''def check(tim):
    lines_seen = set() # holds lines already seen
    outfile = open("C:/Users/NITASHA GUPTA/.spyder-py3/FacialRecognitionAttendanceProject/Attendance Files/Final Attendees.txt", "w")
    for line in open("C:/Users/NITASHA GUPTA/.spyder-py3/FacialRecognitionAttendanceProject/Attendance Files/Attendees.txt", "r"):
        if line not in lines_seen: # not a duplicate
            strr=str(line)
            tim=str(tim)
            if(strr.find(tim)!=-1):
                outfile.write(line)
                lines_seen.add(line)
    outfile.close()'''

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:/Users/NITASHA GUPTA/.spyder-py3/FacialRecognitionAttendanceProject/Trainer/trainer.yml')
cascadePath = "D:/OpenCV/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
font = cv2.FONT_HERSHEY_SIMPLEX     #simply write the text on the image on windows in hershey simplex font
id = 0
names = ['None','Mitanshu','Chota hathi','Nitasha','Kruti','Krunal','Arjun','Tej',]     #our database for names
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height
# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)
aF=open("C:/Users/NITASHA GUPTA/.spyder-py3/FacialRecognitionAttendanceProject/Attendance Files/Attendees.txt","w")


arrayList = []

while True:
    ret, img =cam.read()    #capture the image frame by frame for recognition part
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     #converting the frame into gray scale

    date=datetime.datetime.now()        #to get real os time
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 10,
        minSize = (int(minW), int(minH)),
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)  #arg are img, top left corner, right bottom corner, color of the border and thickness of the line
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        percent=round(100-confidence)
        if (confidence < 100):
            #num=id.
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        
        if percent>=25:
            tim=date.hour
            dat="Date:"+date.strftime("%Y-%m-%d")+" | Time:"+date.strftime("%H:%M")+"-> Name: "+str(id)+"\n"     #+", Percentage Accuracy:"+confidence+"\n"
            #check(tim)
            #aF.write(dat)
            #arrayList.append(dat)
            arrayList.append(str(id))
            print(dat)#+" CHECK:", set(arrayList))
    
    #aF.write(arrayList)
    cv2.imshow('camera',img) 
    
    k = cv2.waitKey(30) & 0xff
    if k== ord('q') or k==27:   #press 'q' key or ESC key
        mySet = set(arrayList)
        
        aF.write(date.strftime("%H:%M :")+str(mySet))
        break

print("\n Exiting Program and cleanup stuff")

cam.release()
cv2.destroyAllWindows()

#aF.write(set(arrayList))

