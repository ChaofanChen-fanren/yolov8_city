from ultralytics import YOLO

model = YOLO("/Users/chenchaofan/python_project/yolov8-city/ultralytics/cfg/models/yolov8-seg-city.yaml",
             task="segmentcity", verbose=True)

model.train()
