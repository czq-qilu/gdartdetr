from ultralytics import RTDETR

# Load a model
model = RTDETR("F:/ultralytics/ultralytics/cfg/models/rt-detr/CARAFE.yaml")  # build a new model from scratch

# Use the model
model.train(data="ultralytics/cfg/datasets/mydata2.yaml", cfg="ultralytics/cfg/default1.yaml", epochs=150)  # train the model