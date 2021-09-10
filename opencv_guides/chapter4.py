import cv2
import numpy as np

# DRAWING SHAPES

img = np.zeros((512, 512, 3), np.uint8)
print(img.shape)

# img[:] = 255, 0, 0

cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (0, 255, 0), 5)

cv2.rectangle(img, (50, 50), (img.shape[1] - 50, img.shape[0] - 50), (0, 0, 255), 5)

cv2.circle(img, (256, 256), 100, (255, 0, 0), 4)

cv2.putText(img, "Text", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

cv2.imshow("Image", img)

cv2.waitKey(0)
