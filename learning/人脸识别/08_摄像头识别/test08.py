import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def face_detect_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detector = cv.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30 ,30))
    for x, y, w, h in faces:
        cv.rectangle(img, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=2)
        cv.circle(img, center=(x+w//2, y+h//2), radius=w//2, color=(0, 255, 0), thickness=2)
    
    cv.imshow('result', img)

    
#cap = cv.VideoCapture('../src/test06_Trim2.mp4')
cap = cv.VideoCapture(0)
#save files
fps = 5
size = (int (cap.get(cv.CAP_PROP_FRAME_WIDTH)), int (cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
videowrite = cv.VideoWriter('../result/result_1.mp4', cv.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

while True:
    flag, frame = cap.read()

    if not flag:
        break

    face_detect_demo(frame)
    videowrite.write(frame)

    if ord('q') == cv.waitKey(10):
        break

cv.destroyAllWindows()
cv.release()