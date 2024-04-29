from roboflow import Roboflow
from ultralytics import YOLO
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO('yolov8s-seg.pt')
model.to(device=device)
print(model.device.type)

results =  model.train(data=r'C:\Users\joaobo\Documents\OilQuantification\testes-1\data.yaml', epochs=2)