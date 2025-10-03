from ultralytics import YOLO

# Load a YOLO model from a pre-trained weights file
model = YOLO("yolo11m.pt")

# Run MODE mode using the custom arguments ARGS (guess TASK)
model.train(model="yolo11m.pt", data="mtg.v7-download.yolov12/data.yaml", epochs=100, imgsz=640,# imgsz=1920, rect=True,
             batch=16, project="mtg_tap", name='initial_try', seed=42, plots=True)