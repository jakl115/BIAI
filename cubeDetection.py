import cv2

cubeCascade = cv2.CascadeClassifier("Resources/cascade.xml")

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)
cam.set(10, -10)

while True :
    success, img = cam.read(0)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cubes = cubeCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in cubes:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        cv2.putText(img, "cube", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


