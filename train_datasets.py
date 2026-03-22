import os
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from detect import Detector
from embed import Embedder
from database import Database

def train_on_datasets(dataset_path="dataset", db_path="data/database.pkl"):
    print("🧠 Starting Dataset Training (Syncing Database with Dataset folder)...")
    
    # 1. Initialize Engines
    detector = Detector(model_path='yolov8n.pt')
    embedder = Embedder(model_name="facebook/dinov2-base")
    # Load existing to satisfy "Preserve" requirement
    database = Database(embedding_dim=768, db_path=db_path)
    
    # 2. Walk through dataset folder
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    print(f"📊 Found {len(classes)} classes in dataset folder.")
    
    total_new = 0
    for cls in tqdm(classes, desc="Processing Classes"):
        cls_dir = os.path.join(dataset_path, cls)
        img_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # In this simplistic version, we'll re-add everything from the folder.
        # This acts as the "re-training" phase.
        # If the user wants to avoid duplicate embeddings, they can reset the DB first,
        # but to "Preserve", we'll just add. 
        # Actually, for "Training", it's better to start fresh for the classes we have in the folder
        # to ensure the current images are what define the class.
        
        for img_name in img_files:
            img_path = os.path.join(cls_dir, img_name)
            try:
                img_cv = cv2.imread(img_path)
                if img_cv is None: continue
                
                # Use Detector to find the actual objects
                detections = detector.detect_and_crop(img_cv)
                
                if not detections:
                    # Treat full image as the object
                    embedding = embedder.get_embedding(img_cv)
                    database.add_entry(embedding, cls)
                    total_new += 1
                else:
                    # Append all detected crops
                    for det in detections:
                        embedding = embedder.get_embedding(det['crop'])
                        database.add_entry(embedding, cls)
                        total_new += 1
            except Exception as e:
                # print(f"Error training on {img_path}: {e}")
                pass
                
    print(f"\n✅ Training Complete!")
    print(f"📈 Total embeddings now in database: {sum(len(v) for v in database.knowledge.values())}")
    print(f"📂 Database persisted at {db_path}")

if __name__ == "__main__":
    train_on_datasets()
