import os
import json
import cv2
import torch
from PIL import Image
from tqdm import tqdm
from utils import ensure_dir, save_json

CROPS_DIR = "../data/final_cropped"
DETECTIONS_JSON = "../data/output_json/detections.json"
FRAMES_DIR = "../data/frames"
OUTPUT_JSON = "../data/output_json/final_cropped/class0_detections.json"

ensure_dir(CROPS_DIR)

with open(DETECTIONS_JSON, "r") as f:
    detections = json.load(f)

class0_detections = []
count = 0
for det in detections:
    if det.get("class_id") != 0:
        continue

    frame_id = det["frame"]
    frame_filename = f"frame_{frame_id}.jpg" if isinstance(frame_id, int) else frame_id
    frame_path = os.path.join(FRAMES_DIR, frame_filename)

    if not os.path.exists(frame_path):
        print(f"[WARN] Frame not found: {frame_path}")
        continue

    img = cv2.imread(frame_path)
    if img is None:
        print(f"[WARN] Failed to load image: {frame_path}")
        continue

    x1, y1, x2, y2 = map(int, det["bbox"])
    crop = img[y1:y2, x1:x2]

    crop_filename = f"class0_crop_{count}.jpg"
    crop_path = os.path.join(CROPS_DIR, crop_filename)
    cv2.imwrite(crop_path, crop)
    count += 1

    det["crop_path"] = crop_path
    class0_detections.append(det)

save_json(class0_detections, OUTPUT_JSON)
print(f"[INFO] Saved {count} crops for class_id = 0 to '{CROPS_DIR}'")
print(f"[INFO] Saved filtered detections to '{OUTPUT_JSON}'")
