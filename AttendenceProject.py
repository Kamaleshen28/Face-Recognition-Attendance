import cv2
import numpy as np
import face_recognition
import os

from datetime import datetime


path = "Resource/ImageAttedence/Photos"
images = []
names = []
myList = os.listdir(path)

for i in myList:
    img = cv2.imread(path+"/"+str(i))
    images.append(img)
    names.append(os.path.splitext(i)[0])

def findEncodings(images):
    encodings = []
    for i in images:
        imgRGB = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        encodings.append(face_recognition.face_encodings(imgRGB)[0])
    return encodings

def markAttendence(name):
    with open(f'Resource/ImageAttedence/Attendence.csv', 'r+') as f:
        myData= f.readlines()
        nameList = []
        for lines in myData:
            entry = lines.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            f.writelines(f'\n{name},{now}')

encodings = findEncodings(images)
print("Encoding Done")

obj = cv2.VideoCapture(0)
obj.set(3, 640)
obj.set(4, 480)
obj.set(10, 200)

while True:
    success, img = obj.read()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurrFrame = face_recognition.face_locations(imgS)
    encodeCurrFrame = face_recognition.face_encodings(imgS, faceCurrFrame)

    for faceEncode, faceLoc in zip(encodeCurrFrame, faceCurrFrame):
        matches = face_recognition.compare_faces(encodings, faceEncode)
        faceDist = face_recognition.face_distance(encodings, faceEncode)
        matchIndex = list(faceDist).index(min(faceDist))

        #print(matchIndex, names[matchIndex])

        if matches[matchIndex]:
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,0),2)
            cv2.rectangle(img, (x1,y2-40), (x2, y2), (0,0,0), cv2.FILLED)
            cv2.putText(img, names[matchIndex], (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            markAttendence(names[matchIndex])


    cv2.imshow("WebCam", img )


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

