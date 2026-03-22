import os, cv2, torch, numpy as np
from src.detector import ObjectDetector
from src.embedder import FeatureEmbedder
from src.search import VectorSearchEngine
from tqdm import tqdm

def populate():
    print("🚀 Initializing FAISS Population (for main.py CLI)...")
    detector = ObjectDetector()
    # Ensure consistency with main.py: use 384-dim
    embedder = FeatureEmbedder(model_name='dinov2_vits14')
    search_engine = VectorSearchEngine(embedding_dim=384)

    dataset_path = "dataset"
    if not os.path.exists(dataset_path):
        print("❌ Dataset folder not found. Skipping population.")
        return

    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    for cls in tqdm(classes, desc="Populating FAISS"):
        cls_dir = os.path.join(dataset_path, cls)
        img_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_name in img_files:
            img_path = os.path.join(cls_dir, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Detect and crop
            detections = detector.detect_objects(img)
            
            if not detections:
                # Fallback: use whole image
                embedding = embedder.get_embedding(img)
                search_engine.add_object(embedding, cls)
            else:
                for det in detections:
                    crop = detector.crop_and_extract(img, det['bbox'])
                    if crop.size > 0:
                        embedding = embedder.get_embedding(crop)
                        search_engine.add_object(embedding, cls)

    print(f"✅ FAISS Index populated with {len(search_engine.metadata)} entries.")

if __name__ == "__main__":
    populate()
