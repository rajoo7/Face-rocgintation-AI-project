# Face-rocgintation-AI-project
In this project the camera fetch the data and save it in the form of array matrix as  0's and 1's form.

import os
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.face.createLBPHFaceRecognizer()
path='/home/priyanshu/Dataset'
def getImageWithId(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L');
        faceNp=np.array(faceImg, 'uint8')
        ID=int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(faceNp)
        print ID
        IDs.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
    return IDs, faces
IDs,faces=getImageWithId(path)
recognizer.train(faces,np.array(IDs))
recognizer.save('recognizer/trainingdata.yml')
cv2.destroyAllWindows()
