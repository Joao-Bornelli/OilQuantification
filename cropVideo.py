import cv2
import os

path = r'C:\Users\joaobo\Videos\FMS K9 TEST#41.mp4'
directory, filename = os.path.split(path)

filename = filename[:-4]+'_Cropped.mp4'
videoName = os.path.join(directory, filename)


cap = cv2.VideoCapture(path)
ret,frame = cap.read()
roi = cv2.selectROI("Select ROI",frame)
cv2.destroyAllWindows()

width,height = roi[2],roi[3]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(videoName,fourcc,60, (width,height))


while True:
    ret,frame = cap.read()
    if ret:
        cropped_frame = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        height, width = frame.shape[:2]
        
        out.write(cropped_frame)

        # Display the frame (optional)
        # cv2.imshow("Frame", cropped_frame)
        key = cv2.waitKey(1)
        if(key == 27):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()