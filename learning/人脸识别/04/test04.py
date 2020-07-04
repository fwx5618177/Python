import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def face_detect_demo():
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier('D:/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt_tree.xml')
    faces = face_detector.detectMultiScale(gray)
    for x, y, w, h in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=2)
        
    cv.imshow('result', img)
    
img = cv.imread('test02.jpg')
face_detect_demo()

cv.waitKey(4000)
cv.destroyAllWindows()