import cv2 as cv
import numpy as np

cap = cv.VideoCapture('nistagmo.mp4')
#cap = cv.VideoCapture(0)

font = cv.FONT_HERSHEY_SIMPLEX
count = 0
vel = 0
upper = (500, 300)
bottom = (200, 200)
#centroid = np.empty(2, 0)

while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv.imshow('frame', frame)

    r = cv.rectangle(frame, upper, bottom, (0, 0, 255), 2, cv.LINE_8)

    roi = frame[300: 200, 500: 200]
    rows, cols, _ = roi.shape

    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (7, 7), 0)
    _, th1 = cv.threshold(gray, 3, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(th1, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv.contourArea(x), reverse=True)

    for cnt in contours:
        count += 1
        (x, y, w, h) = cv.boundingRect(cnt)
        cv.circle(roi, (int((x+(x+w))/2), int((y+(y+h))/2)), 3, (255, 0, 255), -1)
        cv.rectangle(roi, (x, y), (x+w, y+h), (0, 0, 255), 2)

        fps = cap.get(cv.CAP_PROP_FPS)

        if count == 1:
            centerb = np.array([(x+(x+w)), (y+(y+h))])
            cv.putText(roi, str(centerb), (300, 305), font, 1, (0, 255, 0), 2, cv.LINE_AA)
        else:
            center = np.array([(x+(x+w)), (y+(y+h))])
            #np.concatenate((centerb, center), axis=0, out=centroid)
            dist = np.sqrt(abs(centerb[0] - center[0]) ^ 2 + abs(centerb[1] - center[1]) ^ 2)
            cv.putText(roi, str(dist), (300, 305), font, 1, (0, 255, 0), 2, cv.LINE_AA)
            aux = divmod(count, 30)

            if aux[1] == 0:
                centerb = center

            vel = dist / fps
            cv.putText(roi, str(vel), (50, 55), font, 1, (255, 0, 255), 2, cv.LINE_AA)

        break

    cv.imshow('Roi', roi)
    #print(vel)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
