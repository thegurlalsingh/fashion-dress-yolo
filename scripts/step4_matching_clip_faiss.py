import os
import faiss
import torch
import json
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import clip  

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

catalog_dir = "../data/catalog_images"
catalog_embeddings = []
catalog_image_names = []

print("Encoding catalog images...")
for fname in tqdm(os.listdir(catalog_dir)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    path = os.path.join(catalog_dir, fname)
    image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image).cpu().numpy().astype("float32")
    catalog_embeddings.append(embedding)
    catalog_image_names.append(fname)

catalog_matrix = np.vstack(catalog_embeddings)
faiss.normalize_L2(catalog_matrix)

index = faiss.IndexFlatIP(catalog_matrix.shape[1])
index.add(catalog_matrix)

crop_dir = "../data/final_cropped"
results = []

print("Matching frame crops to catalog...")
for fname in tqdm(os.listdir(crop_dir)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    path = os.path.join(crop_dir, fname)
    image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image).cpu().numpy().astype("float32")
    faiss.normalize_L2(embedding)  

    D, I = index.search(embedding, k=1) 
    matched_idx = I[0][0]
    similarity_score = float(D[0][0]) 

    results.append({
        "frame_crop": fname,
        "match": catalog_image_names[matched_idx],
        "product_id": os.path.splitext(catalog_image_names[matched_idx])[0],
        "similarity": round(similarity_score, 4)
    })

output_path = "../data/output_json/match_output.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Saved {len(results)} match results to {output_path}")

