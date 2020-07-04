import cv2 as cv

img = cv.imread('./test01.jpg')

cv.imshow('input image', img)
cv.waitKey(4000)
cv.destroyAllWindows()