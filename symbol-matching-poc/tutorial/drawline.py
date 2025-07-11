import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))

    image = cv2.line(frame,(0,0),(width,height),(255,0,0),10)
    image = cv2.line(image,(0,height),(width,0),(0,255,0),10)
    image = cv2.rectangle(image,(100,100),(200,200),(128,128,128), -1) #square
    image = cv2.circle(image,(100,100),60,(0,0,255), -1) #circle
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.putText(image, "Tim is Great!", (10,height - 10), font, 1, (0,0,0), 5, cv2.LINE_AA )

    cv2.imshow("frame",frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()