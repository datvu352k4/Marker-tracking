import ultralytics
from ultralytics import YOLO

if __name__ == "__main__":
    model=YOLO("yolov8n.pt")
    model.train(
        data="dataset.yaml",
        epochs=100,
        hsv_h=0.02, hsv_s=0.3, hsv_v=0.35,
        translate=0.08, scale=0.3, degrees=5.0, shear=3.0, perspective=0.0005,
        fliplr=0.2, flipud=0.0,
        mosaic=0.15, mixup=0.05, erasing=0.05,
        auto_augment=None,
        imgsz=640,
        device="0",
        workers=4, 
    )
