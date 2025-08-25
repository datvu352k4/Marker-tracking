# json_to_mot_no_dup.py
import json, os

JSON_PATH = "gt.json"
OUT_PATH  = "gt.txt"
VIDEO_W, VIDEO_H = 1280, 720 

def pct2px(x, y, w, h, W, H):
    return x*W/100.0, y*H/100.0, w*W/100.0, h*H/100.0

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)
tasks = data if isinstance(data, list) else [data]

lines = []
written = set() 

tid = 0
for task in tasks:
    for ann in task.get("annotations", []):
        for res in ann.get("result", []):
            if res.get("type") != "videorectangle": 
                continue
            seq = sorted(res["value"]["sequence"], key=lambda s: s["frame"])
            if not seq: 
                continue
            tid += 1
            for i in range(len(seq)-1):
                a, b = seq[i], seq[i+1]
                if not a.get("enabled", True):
                    continue
                fa, fb = int(a["frame"]), int(b["frame"])
                if fb <= fa:
                    continue
                denom = max(fb - fa, 1)
                for f in range(fa, fb):
                    t = (f - fa) / denom
                    x = a["x"] + (b["x"] - a["x"]) * t
                    y = a["y"] + (b["y"] - a["y"]) * t
                    w = a["width"]  + (b["width"]  - a["width"])  * t
                    h = a["height"] + (b["height"] - a["height"]) * t
                    X,Y,W,H = pct2px(x,y,w,h, VIDEO_W, VIDEO_H)
                    key = (f, tid)
                    if key not in written:
                        written.add(key)
                        lines.append(f"{f},{tid},{X:.2f},{Y:.2f},{W:.2f},{H:.2f},1,-1,-1,-1")

            last = seq[-1]
            if last.get("enabled", True):
                fL = int(last["frame"])
                X,Y,W,H = pct2px(last["x"], last["y"], last["width"], last["height"], VIDEO_W, VIDEO_H)
                key = (fL, tid)
                if key not in written:
                    written.add(key)
                    lines.append(f"{fL},{tid},{X:.2f},{Y:.2f},{W:.2f},{H:.2f},1,-1,-1,-1")

lines.sort(key=lambda s: (int(s.split(",")[0]), int(s.split(",")[1])))
with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"OK: wrote {len(lines)} lines -> {OUT_PATH}")
