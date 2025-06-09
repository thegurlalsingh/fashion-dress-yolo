import os
import torch
import clip
from PIL import Image
from tqdm import tqdm
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

vibes = [
    "Coquette",
    "Clean Girl",
    "Cottagecore",
    "Streetcore",
    "Y2K",
    "Boho",
    "Party Glam"
]

text_prompts = [f"This is a photo of a {vibe} outfit" for vibe in vibes]
text_tokens = clip.tokenize(text_prompts).to(device)

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

image_dir = "../data/final_cropped"  
output_file = "../data/output_json/vibe_predictions.json"
results = []

for fname in tqdm(os.listdir(image_dir)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    path = os.path.join(image_dir, fname)
    image = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        image_feature = model.encode_image(image)
        image_feature /= image_feature.norm(dim=-1, keepdim=True)

        similarity = (image_feature @ text_features.T).squeeze(0).cpu().numpy()
        best_idx = int(similarity.argmax())
        best_vibe = vibes[best_idx]
        confidence = float(similarity[best_idx])

    results.append({
        "image": fname,
        "predicted_vibe": best_vibe,
        "confidence": round(confidence, 4)
    })

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Saved vibe predictions for {len(results)} images to {output_file}")
