import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import hog
import cv2 as cv
import os
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import joblib



imgsPath = r'C:\Users\joaobo\Documents\OilVisualization\Images\usar'


imagesList = []
hogList = []
featuresList = []
labelsList = []

def randomRotation(img):
    angle = np.random.randint(-50,50)
    rows,cols,_ = img.shape
    rotation_matrix = cv.getRotationMatrix2D((cols/2,rows/2),angle,1)
    return cv.warpAffine(img,rotation_matrix,(cols,rows),borderMode=cv.BORDER_REPLICATE)
    
def randomFlip(img):
    return cv.flip(img, np.random.randint(-1,2))



imgs = []

for path in tqdm(os.listdir(imgsPath),'Generating new Images:'):

    img = cv.imread(imgsPath + "\\" + path)
    img = cv.resize(img,(250,250))
    
    for i in range(0,6):
        rotated = randomRotation(img)
        imgs.append(rotated)
        labelsList.append(path.lower()[:7])


print('Labels Size: ',len(labelsList))
print('Images Size: ',len(imgs))

for img in tqdm(imgs,'Creating Hogs:'):
    grayimg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
    # fd, hog_image1 = hog(
    #     grayimg,
    #     orientations=3,
    #     pixels_per_cell=(3, 3),
    #     cells_per_block=(1, 1),
    #     visualize=True,
    #     transform_sqrt=True,
    # )
    
    imagesList.append(img)
    # hogList.append(hog_image1)
    hogList.append(grayimg)
    # featuresList.append(fd)


# cv.imshow('hog',hog_image1)
cv.waitKey(250) 
print('images: ',len(imagesList))
print('hog: ',len(hogList))
print('labels: ',len(labelsList))

com = np.hstack(hogList[:14])
sem = np.hstack(hogList[14:])

X_train,X_test_val,y_train,y_test_val = train_test_split(featuresList,labelsList,test_size=0.4,random_state=42)

X_val,X_test,y_val,y_test = train_test_split(X_test_val,y_test_val,test_size=0.5,random_state=42)

# print(X_train.shape[0], X_test.shape[0], X_val.shape[0])
print()
print('Train: ',len(X_train))
print('Test: ',len(X_test))
print('Val: ',len(X_val))

svm_model = SVC(kernel='rbf',C=1.0,probability=True)
svm_model.fit(X_train,y_train)

joblib.dump(svm_model,'model.pkl')


y_pred = svm_model.predict(X_test)
y_val = svm_model.predict(X_val)

print(y_pred.shape)
# cv.imshow('predicted',y_pred)
# cv.waitKey(0)
acc = accuracy_score(y_test,y_pred)
print("Accuracy Test:", acc)
validation = accuracy_score(y_test,y_val)
print("Accuracy Val:", validation)  

