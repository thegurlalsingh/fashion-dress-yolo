import os
import json

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_json(data, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
