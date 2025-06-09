import json
from collections import Counter

with open("../data/output_json/filtered_matches_final.json") as f:
    matches = {item["frame_crop"]: item for item in json.load(f)}

with open("../data/output_json/vibe_predictions.json") as f:
    vibes = {item["image"]: item for item in json.load(f)}

with open("../data/output_json/dress_colors.json") as f:
    colors = {item["image"]: item for item in json.load(f)}

video_id = "reel_001"
vibe_counter = Counter()

all_products = []

for image_name, match in matches.items():
    vibe_info = vibes.get(image_name, {})
    color_info = colors.get(image_name, {})

    vibe = vibe_info.get("predicted_vibe")
    if vibe:
        vibe_counter[vibe] += 1

    product_id = match.get("product_id")
    similarity = match.get("similarity", 0)
    if not product_id:
        continue

    product_data = {
        "color": color_info.get("color_name"),
        "matched_product_id": product_id,
        "match_type": "exact" if similarity >= 0.9 else "similar",
        "confidence": round(similarity, 4)
    }
    all_products.append(product_data)

top_products = sorted(all_products, key=lambda x: x["confidence"], reverse=True)[:3]

output = {
    "video_id": video_id,
    "vibes": [v for v, _ in vibe_counter.most_common(2)],  
    "products": top_products
}

with open("../data/output_json/video_summary.json", "w") as f:
    json.dump(output, f, indent=2)

print("✅ Saved top 3 matches combined video summary to video_summary.json")
