from ctypes import sizeof
from dataclasses import asdict
from pandas import DataFrame
from ultralytics import YOLO 
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
path = r'C:\Users\joaobo\Videos\FMS K9 TEST#41.mp4'
videoName = path.split('\\')[-1][:-4]

cap = cv.VideoCapture(path)

frameNum = 0
acq = False

ret,frame = cap.read()
roi = cv.selectROI("Select ROI",frame)
cv.destroyAllWindows()
model = YOLO(r'C:\Users\joaobo\Documents\OilVisualization\runs\Segment More Images\segment\train2\weights\best.pt')

oilNumber = {'x':[],
             'y':[]}


def showGraph(ax,x,y):
    ax.clear()
    ax.plot(x,y)
    plt.pause(0.01)
    

frameNum = 0

fig, ax = plt.subplots()

while True:
    ret,frame = cap.read()
    if ret:
        frame = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        
        key = cv.waitKey(1)
        if(key == 27):
            break

        prediction = model.predict(frame)
        
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        height, width = frame.shape[:2]
        
        masked = np.zeros((height, width), dtype=np.uint8)
        
        if prediction[0].masks != None:
            for xy in prediction[0].masks: 
                drw = np.array(xy.xy).astype(int)
                masked_bgr = cv.fillConvexPoly(masked,points=drw,color=(255,0,0))
        else:
            masked_bgr = np.zeros((height, width), dtype=np.uint8)
            
        cv.imshow('original',frame)
        cv.imshow('found',masked_bgr)
        mean = np.round(masked_bgr.mean(),3)
        
        oilNumber['x'].append(int((cap.get(cv.CAP_PROP_POS_FRAMES)/60))*1000)
        oilNumber['y'].append(int(mean*1000))

        frameNum += 5
        cap.set(cv.CAP_PROP_POS_FRAMES,frameNum)
    else:
        break

DataFrame(oilNumber).to_csv(videoName + '.csv')

plt.close()
    
cap.release()
cv.destroyAllWindows()
print(oilNumber)


