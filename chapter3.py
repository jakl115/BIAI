import cv2
import numpy as np

img = cv2.imread("Resources/Kofuku.jpg")
print(img.shape)

imgResize = cv2.resize(img, (450, 700))
print(imgResize.shape)

imgCropped = img[0:250, 0:225]

cv2.imshow("Image", imgCropped)

cv2.waitKey(0)