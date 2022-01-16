
import cv2 as cv
import numpy as np
import os
path='./dataset/cropped_image'
manager_names=[]
X=[]
Y=[]
face_detect=cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
for entry in os.scandir(path):
    manager_name=entry.path.split('\\')[-1]
    manager_names.append(manager_name)
    label_name=manager_names.index(manager_name)
    for img in os.scandir(entry.path):
        raw_img=cv.imread(img.path)
        gray=cv.cvtColor(raw_img,cv.COLOR_BGR2GRAY)
        face=face_detect.detectMultiScale(gray,1.3,1)
        for (x,y,w,h) in face:
            face_roi=gray[x:x+w,y:y+h]
            X.append(face_roi)
            Y.append(label_name)
print(manager_names)
X=np.array(X,dtype='object')
Y=np.array(Y)
face_recognizer= cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(X,Y)
face_recognizer.save('face_trained.yml')
np.save('features.npy',X)
np.save('labels.npy',Y)

