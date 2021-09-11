import cv2
import numpy as np

def empty(a):
    pass


trackbarName = "Trackbars"
cv2.namedWindow(trackbarName)
cv2.resizeWindow(trackbarName, 640, 280)
cv2.createTrackbar("Hue min", trackbarName, 34, 179, empty)
cv2.createTrackbar("Hue max", trackbarName, 179, 179, empty)
cv2.createTrackbar("Sat min", trackbarName, 82, 255, empty)
cv2.createTrackbar("Sat max", trackbarName, 161, 255, empty)
cv2.createTrackbar("Val min", trackbarName, 216, 255, empty)
cv2.createTrackbar("Val max", trackbarName, 255, 255, empty)

while True:
    img = cv2.imread("C:/Users/janni/Desktop/ml/BIAI/Resources/Kofuku.jpg")

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue min", trackbarName)
    h_max = cv2.getTrackbarPos("Hue max", trackbarName)
    s_min = cv2.getTrackbarPos("Sat min", trackbarName)
    s_max = cv2.getTrackbarPos("Sat max", trackbarName)
    v_min = cv2.getTrackbarPos("Val min", trackbarName)
    v_max = cv2.getTrackbarPos("Val max", trackbarName)

    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow("Original", img)
    cv2.imshow("HSV", imgHSV)
    cv2.imshow("Mask", mask)
    cv2.imshow("Result", imgResult)
    cv2.waitKey(1)
