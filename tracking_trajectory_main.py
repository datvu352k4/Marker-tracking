import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import os

def distinct_color(track_id: int) -> tuple:
    hue = (track_id * 137) % 360
    saturation = 200
    value = 255
    hsv_color = np.uint8([[[hue / 2, saturation, value]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])

def draw_trajectories(frame, trails, tail_thickness=2):
    for tid, pts in trails.items():
        if len(pts) > 1:
            color = distinct_color(tid)
            for j in range(1, len(pts)):
                cv2.line(frame, (int(pts[j-1][0]), int(pts[j-1][1])),
                                (int(pts[j][0]), int(pts[j][1])), color, tail_thickness)


model = YOLO("Training_data/runs/detect/train/weights/yolov8.engine")  
video_path = "vid.mp4"
output_path = "tracked_output.mp4"
result_txt = "marker_trajectories.txt"
tracker = "bytetrack.yaml"

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

trails = defaultdict(lambda: deque())

if os.path.exists(result_txt):
    os.remove(result_txt)
f = open(result_txt, "a")

frame_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_id += 1
    # YOLO detect + track
    results = model.track(frame, tracker=tracker, persist=True, verbose=False)

    if results and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy() 
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy()

        for box, track_id, conf in zip(boxes, track_ids, confs):
            x1, y1, x2, y2 = box.astype(float) 
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            color = distinct_color(track_id)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"ID:{track_id} | {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)
            trails[track_id].append((cx, cy))
            f.write(f"{frame_id},{track_id},{cx:.4f},{cy:.4f},{conf:.4f}\n")

    draw_trajectories(frame, trails, tail_thickness=2)
    out.write(frame)
    cv2.imshow("YOLOv8 + ByteTrack Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
f.close()
cap.release()
out.release()
cv2.destroyAllWindows()
