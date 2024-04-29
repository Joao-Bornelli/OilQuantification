from turtle import width
from pandas import DataFrame
from ultralytics import YOLO 
import cv2 as cv
import numpy as np

path = r'C:\Users\joaobo\Videos\FMS K9 TEST#41_Cropped.mp4'
videoName = path.split('\\')[-1][:-4]

model = YOLO(r'C:\Users\joaobo\Documents\OilQuantification\runs\runs\segment\train2\weights\best.pt')
model.to('cuda')


prediction = model.predict(path,stream=True)

oilNumber = []
for p in prediction:
    frame = p.orig_img
    height, width = frame.shape[:2]
    masked = np.zeros((height, width), dtype=np.uint8)
    
    if p.masks != None:
        for mask in p.masks.xy:
            masked = cv.fillConvexPoly(masked,points=np.array(mask).astype(int),color=(255,0,0))
    else:
        masked = np.zeros((height, width), dtype=np.uint8)# Masks object for segment masks outputs
    
    mean = int(masked.mean()*1000)
    oilNumber.append(mean)
    cv.imshow("frame",masked)
    key = cv.waitKey(1)
    if(key == ord('q')):
        break

DataFrame(oilNumber).to_csv(videoName + '.csv')
cv.destroyAllWindows()