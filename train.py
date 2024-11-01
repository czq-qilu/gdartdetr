from ultralytics import YOLO

# Load a model
model = YOLO('ultralytics/cfg/models/v8/yolov8_gdafpn.yaml')  # build a new model from YAML
model = YOLO('yolov8.yaml') # build from YAML and transfer weights

# Train the model
model.train(data='ultralytics/cfg/datasets/mydata3.yaml', epochs=150, imgsz=640)



