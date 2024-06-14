import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt

# from roboflow import Roboflow
# rf = Roboflow(api_key="B6j5ifiLvo5a3MpAHSoS")
# project = rf.workspace("oilvisualization").project("testes-zaytx")
# # version = project.version(5)
# version = project.version(3)
# dataset = version.download("yolov8")



iouPred = []
iouTrue = []
iou = []

imagesPath = r'C:\Users\joaobo\Documents\OilVisualization\testes-3\valid\images'
imagesArray = os.listdir(imagesPath)

masksPath = r'C:\Users\joaobo\Documents\OilVisualization\testes-3\valid\labels'

masksArray = os.listdir(masksPath)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO(r'C:\Users\joaobo\Documents\OilVisualization\runs\Segment More Images\segment\train2\weights\best.pt').to(device=device)

def parse_data(data, width, height):
    lines = data.strip().split('\n')
    polygons = []
    for line in lines:
        points = line.split()[1:]
        coords = [(float(points[i * 2 + 1]) * width, float(points[i * 2]) * height) for i in range(len(points) // 2)]
        polygons.append(np.array(coords, dtype=np.int32))
    return polygons

for i, imageName in enumerate(imagesArray):
    image = cv2.imread(os.path.join(imagesPath,imageName))
    blackMask = np.zeros((image.shape[0], image.shape[1]))

    results = model.predict(image, conf=0.2)

    maskPath = os.path.join(masksPath,imageName[:-4] + '.txt')
    imagePath = os.path.join(imagesPath,imageName)

    if os.path.exists(maskPath):
        with open(maskPath, 'r') as file:
            data = file.read()

        polygons = parse_data(data, image.shape[1], image.shape[0])
        mask = np.zeros((image.shape[1], image.shape[0]), dtype=np.uint8)

        for polygon in polygons:
            cv2.fillPoly(mask, [polygon], color=255)

        mask_corrected = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        mask_corrected = cv2.flip(mask_corrected, 1)
    else:
      continue

    predicted_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Assuming `results` contains the segmentation data
    for result in results[0].masks:
      for segment in result.xy:
        polygon = np.array(segment, dtype=np.int32)
        cv2.fillPoly(predicted_mask, [polygon], color=255)


    intersection = np.logical_and(predicted_mask,mask_corrected)
    union = np.logical_or(predicted_mask,mask_corrected)
    iou.append(np.sum(intersection)/np.sum(union))

    size = predicted_mask.size
    true = np.sum(predicted_mask == 255)
    iouPred.append(true/size)


    size = mask_corrected.size
    true = np.sum(mask_corrected == 255)
    iouTrue.append(true/size)



    if(i >= 40):
      break

fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True) 

ax[0].plot(iou)
ax[0].set_title("IOU")
ax[0].set_xlabel("Images")
ax[0].set_ylabel("IOU Score")

ax[1].plot(iouTrue, label='True Value')
ax[1].plot(iouPred, label='Pred. Value')

ax[1].set_title("Quantification")
ax[1].set_xlabel("Images")
ax[1].set_ylabel("'Oils' Detected")
ax[1].legend()



mse = [(true-pred)**2 for true,pred in zip(iouTrue,iouPred)]
mse = np.round(np.mean(sum(mse)),4)
rmse = np.round(np.sqrt(mse),4)
mae = np.round(np.mean(np.sum([(true - pred) for true,pred in zip(iouTrue,iouPred)])),4)



print(mse)
print(rmse)
print(mae)


plt.tight_layout()
plt.show()