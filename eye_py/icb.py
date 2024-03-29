import numpy as np
import cv2 as cv
img = cv.imread('olhos3.jpg',0)
img = cv.medianBlur(img,5)
cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)

width, height, channels = cimg.shape
if width >800 or height >800:
    cimg = cv.resize(cimg, None, fx=.5, fy=.5, interpolation = cv.INTER_CUBIC)
    img = cv.resize(img, None, fx=.5,fy=.5, interpolation = cv.INTER_CUBIC)

circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,img.shape[0],
                            param1=50,param2=30,minRadius=60,maxRadius=80)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
cv.imshow('detected circles',cimg)
cv.waitKey(0)
cv.destroyAllWindows()
