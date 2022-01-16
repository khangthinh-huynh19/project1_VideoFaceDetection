import cv2 as cv
import numpy as np
face_detect=cv.CascadeClassifier(cv.data.haarcascades+"haarcascade_frontalface_default.xml")
manager_name=['lampard','hansi flick','jurgen klopp','mourinho','pep guardiola']

#Load trained model
face_recognizer=cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

#Load trained features and labels
features=np.load('features.npy',allow_pickle=True)
labels=np.load('labels.npy')

video=cv.VideoCapture('managerdetection.mp4')
while True:
    isTrue,frame=video.read()
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    face=face_detect.detectMultiScale(gray,1.3,1)
   
    for (x,y,w,h) in face:
        #Remember the size of face_roi is (height,width)
        face_roi=gray[y:y+h,x:x+w,]
        label,confidence = face_recognizer.predict(face_roi)
        cv.putText(frame, str(manager_name[label]), (50,50), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    cv.imshow('Test',frame)
    if cv.waitKey(20) & 0xFF==ord(' '):
        break

video.release()
video.destroyAllWindows()
cv.waitKey(0)