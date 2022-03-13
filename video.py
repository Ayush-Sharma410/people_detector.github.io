import cv2

capture=cv2.VideoCapture(0)
capture.set(3,640)
capture.set(4,480)
while True:
    isTrue, frame=capture.read()
    cv2.imshow('carvideo',frame)

    if cv2.waitKey(1) & 0xFF==ord('d'):
        break
capture.release()
cv2.destroyAllWindows()