import cv2 as cv
import numpy as np

cap = cv.VideoCapture('nistagmo.mp4')

font = cv.FONT_HERSHEY_SIMPLEX

while cap.isOpened():
    _, frame = cap.read()
    (x, y, w, h) = cv.selectROI('roi', frame)
    cropped = frame[y:y+h, x:x+w]
    cv.imshow('Roi', cropped)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()