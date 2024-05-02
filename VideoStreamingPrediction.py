from turtle import width
from pandas import DataFrame
from ultralytics import YOLO 
import cv2 as cv
import numpy as np
from movingAvg import movingAVG

videoPath = r'C:\Users\joaobo\Videos\No oil_Cropped.mp4'
videoName = videoPath.split('\\')[-1][:-4]


model = YOLO(r'C:\Users\joaobo\Documents\OilQuantification\best.pt')


#choosing a medium size yolo pretrained model and loading the weights from my training
# model = YOLO('yolov8m-seg.yaml').load('path to weights.pt')
model.to('cuda')

prediction = model.predict(videoPath,stream=True,conf = 0.2)

oilNumber = []
for p in prediction:
    frame = p.orig_img
    height, width = frame.shape[:2]
    masked = np.zeros((height, width), dtype=np.uint8)
    
    if p.masks != None:
        for mask in p.masks.xy:
            masked = cv.fillConvexPoly(masked,points=np.array(mask).astype(int),color=(255,0,0))
    else:
        masked = np.zeros((height, width), dtype=np.uint8)
    
    mean = int(masked.mean()*1000)
    oilNumber.append(mean)
    mask = cv.threshold(masked, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    frame[mask==255] = (36,255,12)
    cv.imshow("frame",frame)
    key = cv.waitKey(1)
    
    if(key == ord('q')):
        break

DataFrame(oilNumber).to_csv(videoName + '.csv')
movingAVG(videoName + '.csv')
cv.destroyAllWindows()