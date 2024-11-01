from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s.pt')  # load an official model
model = YOLO('F:/ultralytics/runs/lens/yolov8/weights/best.pt')  # load a custom model

# Validate the model
metrics = model.val(data="ultralytics/cfg/datasets/mydata2.yaml", imgsz=640, split='val', batch=1, conf=0.001, iou=0.6, name='run', optimizer='Adam')

print(f"mAP50-95: {metrics.box.map}")  # map50-95
print(f"mAP50: {metrics.box.map50}")  # map50
print(f"mAP75: {metrics.box.map75}")  # map75
speed_metrics = metrics.speed
total_time = sum(speed_metrics.values())
fps = 1000 / total_time
print(f"FPS: {fps}")  # FPS