import os
import time
import urllib.request
import numpy as np
import cv2
from duckduckgo_search import DDGS
from detect import Detector
from embed import Embedder
from database import Database

# Target Classes to Expand (50+ classes for wide recognition range)
NEW_CLASSES = [
    "keyboard", "mouse", "mobile phone", "book", "cup", 
    "pen", "monitor", "shoes", "watch", "glasses",
    "car", "bicycle", "umbrella", "sports ball", "wine glass",
    "suitcase", "cat", "motorcycle", "clock", "handbag",
    "backpack", "bottle", "keyboard", "laptop", "mouse", 
    "person", "chair", "sofa", "dining table", "potted plant",
    "bed", "toilet", "tv", "laptop", "mouse", "remote", 
    "microwave", "oven", "toaster", "sink", "refrigerator", 
    "book", "clock", "vase", "scissors", "teddy bear", 
    "hair drier", "toothbrush", "apple", "banana", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "dog", "horse", "sheep", "cow", "elephant", "bear", 
    "zebra", "giraffe", "traffic light", "fire hydrant", 
    "stop sign", "parking meter", "bench", "bird", "boat"
]

def expand_knowledge():
    print("🚀 Initializing Knowledge Expansion Engine...")
    
    # Initialize components (using existing logic)
    detector = Detector(model_path='yolov8n.pt')
    embedder = Embedder(model_name="facebook/dinov2-base")
    # Load existing database (it automatically loads matching database.pkl)
    database = Database(embedding_dim=768, db_path='data/database.pkl')
    
    start_count = len(database.labels)
    print(f"🔍 Current classes in database: {database.labels} (Total: {start_count})")

    with DDGS() as ddgs:
        for cls in NEW_CLASSES:
            if cls in database.labels:
                print(f"⏩ Class '{cls}' already exists, skipping/refreshing...")
                
            print(f"✨ Processing Class: {cls}...")
            os.makedirs(f"dataset/{cls}", exist_ok=True)
            
            try:
                # Search for clear images
                results = ddgs.images(
                    keywords=f"{cls} object clear solo",
                    region="wt-wt",
                    safesearch="moderate",
                    max_results=7
                )
            except Exception as e:
                print(f"⚠️ Search limit reached for '{cls}', waiting 5s...")
                time.sleep(5)
                continue

            added_count = 0
            for i, r in enumerate(results):
                if added_count >= 5: break # Limit 5 for lightweight footprint
                
                url = r.get("image")
                if not url: continue
                
                img_path = f"dataset/{cls}/{i}.jpg"
                try:
                    # Download
                    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                    img_bytes = urllib.request.urlopen(req, timeout=5).read()
                    
                    # Convert to CV2
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img_cv is not None:
                        # Save image to dataset folder
                        with open(img_path, "wb") as f:
                            f.write(img_bytes)
                            
                        # Extract Object (localization only)
                        detections = detector.detect_and_crop(img_cv)
                        
                        if not detections:
                            # Fallback: Entire image check
                            embedding = embedder.get_embedding(img_cv)
                            database.add_entry(embedding, cls)
                            added_count += 1
                        else:
                            # Use largest detection crop
                            best_det = max(detections, key=lambda d: (d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1]))
                            embedding = embedder.get_embedding(best_det['crop'])
                            database.add_entry(embedding, cls)
                            added_count += 1
                except:
                    continue
            
            print(f"✅ Added {added_count} samples for '{cls}'.")
            time.sleep(2) # Throttle to respect DDG

    final_count = len(database.labels)
    print(f"\n🎉 Expansion Complete!")
    print(f"📊 Previous classes: {start_count}")
    print(f"📊 New classes count: {final_count}")
    print(f"📦 Total embeddings saved in data/database.pkl")

if __name__ == "__main__":
    expand_knowledge()
