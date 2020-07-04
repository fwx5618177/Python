import cv2 as cv

img = cv.imread('test01.jpg')
x, y, w, h = 50, 50, 80, 80
cv.rectangle(img, (x, y, x+w, y+h), color=(0, 255, 0), thickness=2)
cv.imshow('rectange',  img)
cv.circle(img, center=(x+w//2, y+h//2), radius=w//2, color=(0, 0, 255), thickness=2)

cv.imshow('result image', img)
cv.waitKey(0)
cv.destroyAllWindows()