import cv2
import face_recognition
import numpy as np


imgbja= face_recognition.load_image_file("bja.jpg")
imgbja=cv2.cvtColor(imgbja,cv2.COLOR_BGR2RGB)
imggreenday= face_recognition.load_image_file("greenday1.jpg")
imggreenday=cv2.cvtColor(imggreenday,cv2.COLOR_BGR2RGB)

facelocation  = face_recognition.face_locations(imgbja)[0]
encodebja = face_recognition.face_encodings(imgbja)[0]
#print(facelocation)
cv2.rectangle(imgbja,(facelocation[3],facelocation[0]),(facelocation[1],facelocation[2]),(225,0,225),2)

facelocation_test  = face_recognition.face_locations(imggreenday)[0]
encodegreenday = face_recognition.face_encodings(imggreenday)[0]
#print(facelocation)
cv2.rectangle(imggreenday,(facelocation_test[3],facelocation_test[0]),(facelocation_test[1],facelocation_test[2]),(225,0,225),2)

results = face_recognition.compare_faces([encodegreenday],encodebja)
print(results)
cv2.imshow("billie",imgbja)
cv2.imshow("greenday",imggreenday)
cv2.waitKey(0)