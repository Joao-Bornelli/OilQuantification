import cv2 as cv
import numpy as np
from skimage.feature import hog

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

# predictPath = r'C:\Code\OilVisualization\Images\usar\SemOleo (1).jpg' #Sem

predictPath = r'C:\Users\joaobo\Documents\OilVisualization\Images\usar\ComOleo (1).jpg'

img = cv.imread(predictPath)
img = cv.resize(img,(250,250))

grayimg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

print (grayimg.shape)

fd, hog_image1 = hog(
        grayimg,
        orientations=3,
        pixels_per_cell=(3, 3),
        cells_per_block=(1, 1),
        visualize=True,
        transform_sqrt=True,
    )


cv.imshow('img',img)
cv.imshow('hog',hog_image1)
cv.waitKey(0)


fd = fd.reshape(1, -1)

svm_model = joblib.load('model.pkl')


prediction  = svm_model.predict(fd)
pred_seg = prediction.reshape([128,128])

plt.imshow(img.astype(np.uint8))
plt.imshow(pred_seg.astype(np.uint8))
plt.show()
# print(prediction)
# print(decision_scores)