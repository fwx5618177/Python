import os
import cv2
import sys
from PIL import Image
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def getImageAndLabels(path):
    faceSamples = []
    ids = []
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int (os.path.split(imagePath)[-1].split(".")[0])

        faces = face_detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y: y+h, x: x+w])
            ids.append(id)

    return faceSamples, ids


if __name__ == '__main__':
    #get image and Id tag
    path = './data/jm/'
    faces, ids = getImageAndLabels(path)

    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer/trainer.yml')