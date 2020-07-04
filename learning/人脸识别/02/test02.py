#图片灰度
import cv2 as cv
src = cv.imread('./test01.jpg')
cv.imshow('./test01.jpg', src)

gray_img = cv.cvtColor(src, code = cv.COLOR_BGR2GRAY)
cv.imshow('gray_image', gray_img)

cv.imwrite('test02.jpg', gray_img)
cv.waitKey(4000)
cv.destroyAllWindows()