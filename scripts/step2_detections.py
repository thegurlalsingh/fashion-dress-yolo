from ultralytics import YOLO
import cv2
import os
import json

def run_detection(frames_dir="../data/frames", output_json="../data/output_json/detections.json", model_path="yolov8n.pt", num_frames=40):
    model = YOLO(model_path)
    detections = []

    for i in range(num_frames):  
        frame_path = os.path.join(frames_dir, f"frame_{i}.jpg")
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"[WARN] Could not read {frame_path}")
            continue

        results = model(frame)
        boxes = results[0].boxes

        for box in boxes:
            detection = {
                "frame": f"frame_{i}.jpg",
                "class_id": int(box.cls[0]),  
                "confidence": float(box.conf[0]),  
                "bbox": box.xyxy[0].tolist()  
            }
            detections.append(detection)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    print(f"[INFO] Saving {len(detections)} detections to {output_json}")

    with open(output_json, "w") as f:
        json.dump(detections, f, indent=2)


def main():
    run_detection()

if __name__ == "__main__":
    main()
