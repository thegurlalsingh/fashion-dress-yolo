from PIL import Image
from collections import Counter
import numpy as np
from sklearn.cluster import KMeans
import os
from tqdm import tqdm
import json

def rgb_to_name(rgb):
    basic_colors = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "gray": (128, 128, 128),
        "red": (255, 0, 0),
        "green": (0, 128, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "purple": (128, 0, 128),
        "pink": (255, 192, 203),
        "brown": (165, 42, 42),
        "orange": (255, 165, 0),
        "beige": (245, 245, 220)
    }

    distances = {name: np.linalg.norm(np.array(rgb) - np.array(code)) for name, code in basic_colors.items()}
    return min(distances, key=distances.get)

def get_dominant_color(image_path, k=3):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((100, 100))  
    pixels = np.array(image).reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, random_state=42).fit(pixels)
    most_common = Counter(kmeans.labels_).most_common(1)[0][0]
    dominant_rgb = kmeans.cluster_centers_[most_common]
    dominant_rgb = tuple(int(x) for x in dominant_rgb)  
    return dominant_rgb, rgb_to_name(dominant_rgb)

crop_dir = "../data/final_cropped"
output_file = "../data/output_json/dress_colors.json"
results = []

for fname in tqdm(os.listdir(crop_dir)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    path = os.path.join(crop_dir, fname)
    try:
        rgb, color_name = get_dominant_color(path)
        results.append({
            "image": fname,
            "dominant_rgb": list(rgb), 
            "color_name": color_name
        })
    except Exception as e:
        print(f"[ERROR] Failed on {fname}: {e}")

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Saved color predictions for {len(results)} images to {output_file}")
