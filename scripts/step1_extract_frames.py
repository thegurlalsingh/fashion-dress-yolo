import os
import cv2
import json

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_json(data, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def extract_frames(video_path, interval=15, save_dir="../data/frames", metadata_path="../data/output_json/frames_metadata.json"):
    ensure_dir(save_dir)
    cap = cv2.VideoCapture(video_path)

    frames = []
    i = 0
    frame_id = 0

    print("[INFO] Extracting frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % interval == 0:
            frame_path = os.path.join(save_dir, f"frame_{frame_id}.jpg")
            cv2.imwrite(frame_path, frame)
            frames.append({"id": frame_id, "path": frame_path})
            frame_id += 1
        i += 1

    cap.release()
    save_json(frames, metadata_path)
    print(f"[INFO] Extracted {frame_id} frames. Metadata saved to {metadata_path}")

if __name__ == "__main__":
    video_file = "../data/video.mp4"  
    extract_frames(video_file)