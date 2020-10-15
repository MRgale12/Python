import cv2
from matplotlib import pyplot as plt
import numpy as np

cap = cv2.VideoCapture(0)  # Webcam Capture
train_img = cv2.imread('one.jpg')
train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(1000,1.2)


while (True):

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    queryKeypoints, queryDescriptors = orb.detectAndCompute(gray, None)
    trainKeypoints, trainDescriptors = orb.detectAndCompute(train_img_bw, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(queryDescriptors, trainDescriptors)
    matches = sorted(matches, key=lambda val: val.distance)
    final_img = cv2.drawMatches(frame, queryKeypoints,train_img, trainKeypoints, matches[:10], None)
    final_img = cv2.resize(final_img, (1000, 650))

    # Show the final image
    cv2.imshow("Matches", final_img)
    # cv2.imshow('Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
