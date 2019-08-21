import cv2 as cv
import numpy as np

cap = cv.VideoCapture('eye_recording.flv')
#cap = cv.VideoCapture(0)

font = cv.FONT_HERSHEY_SIMPLEX
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    roi = frame
    rows, cols, _ = roi.shape

    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (7, 7), 0)
    _,th1 = cv.threshold(gray, 3, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(th1, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE) 
    contours = sorted(contours, key = lambda x: cv.contourArea(x), reverse = True)

    for cnt in contours:
        count += 1
        (x, y, w, h) = cv.boundingRect(cnt)
        cv.circle(roi, (int((x+(x+w))/2), int((y+(y+h))/2)), 3, (255, 0, 255), -1)
        cv.rectangle(roi, (x, y), (x+w, y+h), (0, 0, 255), 2)

        if(count == 1):
            centerb = [(x+(x+w)), (y+(y+h))]
            cv.putText(roi, str(centerb), (300,305), font, 1, (0,255,0),2,cv.LINE_AA)
        else:
            center = [(x+(x+w)), (y+(y+h))]
            center = np.subtract(center,centerb)
            dist = np.sqrt((centerb[0] - center[0]) ^ 2 + (centerb[1] - center[1]) ^ 2)
            cv.putText(roi, str(dist), (300, 305), font, 1, (0, 255, 0), 2, cv.LINE_AA)
        break

    cv.imshow('Roi', roi)
    cv.imshow('Binary Roi', th1)
    if cv.waitKey(25) == 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
