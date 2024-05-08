from roboflow import Roboflow

rf = Roboflow(api_key="")
project = rf.workspace("oilvisualization").project("testes-zaytx")
version = project.version(1)
dataset = version.download("yolov8")
