import cv2
import numpy as np

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture('nistagmo.mp4')
ddepth = cv2.CV_16S

# Check if camera opened successfully
if cap.isOpened() == False:
    print("Error opening video stream or file")

# Read until video is completed
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        roi = frame[200:350, 0:600]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.Laplacian(gray, ddepth, 3)

        gray = np.float32(gray)
        dst = cv2.cornerHarris(gray, 2, 1, 0.04)

        #rows, cols = gray.shape
        dst = cv2.dilate(dst, None)
        for
        #cv2.circle(roi, int(dst.max()), int(dst.min()), 3, (255, 0, 255), -1)

        roi[dst > 0.01*dst.max()] = [0, 0, 0]
        _, th = cv2.threshold(roi, 3, 255, cv2.THRESH_BINARY_INV)

        # Display the resulting frame
        cv2.imshow('Test', th)
        cv2.imshow('ROI', roi)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()