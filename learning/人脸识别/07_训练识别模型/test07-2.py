import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')


img = cv2.imread('./data/jm_predict/02.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_detecor = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_detecor.detectMultiScale(gray)

for x, y, w, h in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    id, confidence = recognizer.predict(gray[y: y+h, x: x+w])
    print('id:', id, '置信评分:', confidence)

cv2.imshow('result', img)
cv2.waitKey(4000)
cv2.destroyAllWindows()