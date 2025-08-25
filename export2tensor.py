from ultralytics import YOLO

model=YOLO("Training_data/runs/detect/train/weights/yolov8.pt").export(
    format="engine",
    device=0,
    int8=False,     
    half=True,
    dynamic=False,
    batch=1,
    imgsz=(384,640),
    workspace=4,
    data=("Training_data/dataset.yaml")
)
