
# Smart Tagging & Vibe Classification Engine

This repository processes fashion-related videos to extract frames, detect clothing items, match them with products, extract colors, transcribe audio, classify the overall "vibe," and generate a final JSON output for each video.

## Deployment

Don't forget to put video file in data folder.

Run

```bash
  pip install -r requirements.txt
```
requirements.txt should contain-:

```bash
    torch>=2.0.0
    transformers>=4.40.0
    ultralytics>=8.0.200
    opencv-python
    moviepy
    numpy<2.0  # Required due to compatibility issues with Ultralytics
    Pillow
    scikit-learn
    tqdm
```

then run 
```bash
    cd scripts
```
In scripts folder, you have to run each file seprately
To run each file, execute following commands-:

```bash
    python step1_extract_frames.py  
    python step2_detections.py 
    python step3_matching_basic.py 
    python step4_matching_clip_faiss.py 
    python step5_filter_match.py  
    python step6_color.py    
    python step7_vibe_classification.py 
    step8_main.py 
```

Each script will generate its own .json output and save it in data/output_json/, and main.py will combine them into a final result and give video_summary.json.