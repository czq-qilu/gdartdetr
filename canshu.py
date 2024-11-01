from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(r'ultralytics/cfg/models/v8/yolov8_gdafpn.yaml')  # build a new model from YAML
    model.info()

