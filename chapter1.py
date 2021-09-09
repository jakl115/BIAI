import cv2
print("Package Imported")

#img = cv2.imread("Resources/Kofuku.jpg")

#cv2.imshow("Output", img)

#vid = cv2.VideoCapture("Resources/video.mp4")

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
cam.set(10, -10)

while True:
    success, img = cam.read()
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
