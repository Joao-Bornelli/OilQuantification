import numpy as np
import cv2 as cv

videoPath = r"C:\Users\joaobo\Downloads\Test #17.mp4"

imagesPath = r'C:\Users\joaobo\Documents\OilVisualization\Images\AllResized'


cap = cv.VideoCapture(videoPath)

# print(cap.get(cv.CAP_PROP_FPS))

if not cap.isOpened():
    print('error')
else:
    ret, frame = cap.read()
    roi = cv.selectROI(frame)
    frameNum = 0
    while True:
        ret, frame = cap.read()
        if(ret):
            frameNum += 5
            cap.set(cv.CAP_PROP_POS_FRAMES,frameNum)
            frame = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
            cv.imshow('frame',frame)    
            if cv.waitKey(1) == ord(r'q'):
                cap.release()
                break
            cv.imwrite(imagesPath +'\Test #17_'+str(frameNum)+'.jpg',frame)
        else:
            break
