from roboflow import Roboflow
from ultralytics import YOLO
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
model = YOLO('yolov8s-seg.pt')
print(model.device.type)

results = model.train(data=r'C:/Users/joaobo/Documents/OilVisualization/OilDetection-4/data.yaml', epochs=2, batch = 4, device=0)