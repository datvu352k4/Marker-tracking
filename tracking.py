import ultralytics
from ultralytics import YOLO

if __name__ == "__main__":
    model=YOLO("Training_data/runs/detect/train/weights/yolov8.engine")
    results = model.track(source="vid.mp4", 
                          tracker="bytetrack.yaml",
                          save=True, 
                          save_txt=True,
                          show_conf=False,
                          persist=True )