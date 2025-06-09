import json

input_file = "../data/output_json/match_output.json"
output_file = "../data/output_json/filtered_matches_final.json"
similarity_threshold = 0.75  

with open(input_file, "r") as f:
    matches = json.load(f)

filtered = [m for m in matches if m["similarity"] >= similarity_threshold]

with open(output_file, "w") as f:
    json.dump(filtered, f, indent=2)

print(f"✅ Kept {len(filtered)} of {len(matches)} matches with similarity ≥ {similarity_threshold}")
print(f"📄 Saved to: {output_file}")
