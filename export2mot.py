
import argparse
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

def tlwh_from_xyxy(xyxy):
    x1, y1, x2, y2 = map(float, xyxy)
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="vid.mp4")
    ap.add_argument("--weights", default="Training_data/runs/detect/train/weights/best.pt")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--out", default="results_mot.txt")
    ap.add_argument("--conf_min", type=float, default=0.25)
    ap.add_argument("--min_area", type=float, default=50.0)
    ap.add_argument("--classes", nargs="*", type=int)
    ap.add_argument("--target_fps", type=float, default=25.0, help="Mục tiêu fps để ghi .txt")
    ap.add_argument("--keep_orig_index", action="store_true",
                    help="Ghi số frame gốc (ví dụ 1,13,25...) thay vì 1,2,3...")
    args = ap.parse_args()

    cap = cv2.VideoCapture(0 if args.source.isdigit() else args.source)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()

    stride = 1
    if src_fps > 0 and args.target_fps > 0:
        stride = max(1, int(round(src_fps / args.target_fps)))
    print(f"[Info] src_fps={src_fps:.3f}, target_fps={args.target_fps}, vid_stride={stride}")

    model = YOLO(args.weights)

    results_iter = model.track(
        source=args.source,
        tracker=args.tracker,
        stream=True,
        persist=True,
        verbose=False,
        save=False,     
        show=False,
        save_txt=False,
        vid_stride=stride,   
    )

    processed_idx = 0  
    lines = []
    for r in results_iter:
        processed_idx += 1
        if args.keep_orig_index:
            mot_frame_idx = 1 + (processed_idx - 1) * stride
        else:
            mot_frame_idx = processed_idx

        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes = r.boxes
        ids = boxes.id
        if ids is None:
            continue

        ids  = ids.cpu().numpy().astype(np.int64)
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls  = boxes.cls.cpu().numpy().astype(np.int64)

        H, W = r.orig_shape if hasattr(r, "orig_shape") else (None, None)

        keep_idx = []
        for i in range(len(ids)):
            if conf[i] < args.conf_min:
                continue
            if args.classes and cls[i] not in args.classes:
                continue
            x, y, w, h = tlwh_from_xyxy(xyxy[i])
            if w * h < args.min_area:
                continue
            if W is not None and H is not None:
                if x >= W or y >= H or (x + w) <= 0 or (y + h) <= 0:
                    continue
            keep_idx.append(i)

        if not keep_idx:
            continue

        for i in keep_idx:
            x, y, w, h = tlwh_from_xyxy(xyxy[i])
            lines.append(f"{mot_frame_idx},{int(ids[i])},{x:.1f},{y:.1f},{w:.1f},{h:.1f},"
                         f"{float(conf[i]):.4f},-1,-1,-1\n")

    Path(args.out).write_text("".join(lines), encoding="utf-8")
    print(f"Saved {len(lines)} lines to {args.out}")

if __name__ == "__main__":
    main()
