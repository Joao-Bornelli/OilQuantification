from turtle import width
from pandas import DataFrame
from ultralytics import YOLO 
import cv2 as cv
import numpy as np
import cropVideo

path = r'C:\Users\joaobo\Videos\No oil_Cropped.mp4'
videoName = path.split('\\')[-1][:-4]

model = YOLO(r'C:\Users\joaobo\Documents\OilQuantification\runs\runs\segment\train2\weights\best.pt')
model.to('cuda')

prediction = model.predict(path,stream=True,conf = 0.2)

oilNumber = []
for p in prediction:
    frame = p.orig_img
    height, width = frame.shape[:2]
    masked = np.zeros((height, width), dtype=np.uint8)
    
    if p.masks != None:
        for mask in p.masks.xy:
            masked = cv.fillConvexPoly(masked,points=np.array(mask).astype(int),color=(255,0,0))
            # masked = cv.fillPoly(masked, pts=[np.array(mask).astype(int)], color=(255,0,0))
    else:
        masked = np.zeros((height, width), dtype=np.uint8)# Masks object for segment masks outputs
    
    mean = int(masked.mean()*1000)
    oilNumber.append(mean)
    # cv.imshow("frame",masked)

    mask = cv.threshold(masked, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    
    frame[mask==255] = (36,255,12)
    
    cv.imshow("frame",frame)
    
    
    key = cv.waitKey(1)
    if(key == ord('q')):
        break

DataFrame(oilNumber).to_csv(videoName + '.csv')

cropVideo.movingAVG(videoName + '.csv')
cv.destroyAllWindows()