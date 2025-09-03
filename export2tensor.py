from ultralytics import YOLO

model=YOLO("Training_data/runs/detect/train/weights/640model.pt").export(
    format="engine",
    device=0,
    int8=False,     
    half=True,
    dynamic=False,
    batch=1,
    imgsz=(640,640),
    workspace=4,
    data=("Training_data/dataset.yaml")
)
