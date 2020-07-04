import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def face_detect_demo():
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier('D:/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_alt_tree.xml')
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.008, minNeighbors=4, maxSize=(47, 47), minSize=(28 ,28))
    for x, y, w, h in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=2)
        cv.circle(img, center=(x+w//2, y+h//2), radius=w//2, color=(0, 255, 0), thickness=2)
    
    cv.imshow('result', img)

img = cv.imread('test05.jpg')

face_detect_demo()
cv.waitKey(4000)
cv.destroyAllWindows()