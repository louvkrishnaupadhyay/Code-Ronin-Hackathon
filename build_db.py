import os
import glob
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from detect import Detector
from embed import Embedder
from database import Database

def is_blurry(image, threshold=100.0):
    """
    Computes the Laplacian variance of the image to determine blurriness.
    A lower variance means the image is blurrier.
    """
    if isinstance(image, Image.Image):
        # Convert PIL to CV2 grayscale
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    elif len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    return variance < threshold

def build_database(dataset_dir="dataset", db_dir="data", use_detector=False):
    """
    Processes a directory structured as dataset/class_name/images.jpg
    Builds the vector database to be used by the CodeNova adaptive vision system.
    """
    if not os.path.exists(dataset_dir):
        print(f"Dataset directory '{dataset_dir}' not found. Please structure your data as:")
        print(f"  {dataset_dir}/<class_name>/<image.jpg>")
        return

    print("Initializing components for database building...")
    detector = Detector(model_path="yolov8n.pt") if use_detector else None
    embedder = Embedder(model_name="facebook/dinov2-base")
    db_path = os.path.join(db_dir, "database.pkl")
    database = Database(embedding_dim=768, db_path=db_path)
    
    # We clear existing DB since we are building fresh
    print("Clearing existing database...")
    if os.path.exists(database.db_path): os.remove(database.db_path)
    database = Database(embedding_dim=768, db_path=db_path)

    classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    print(f"Found {len(classes)} classes: {classes}")
    
    total_images_processed = 0
    rejected_images = 0

    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        # Find images 
        image_paths = glob.glob(os.path.join(cls_dir, "*.jpg")) + glob.glob(os.path.join(cls_dir, "*.png"))
        
        print(f"\nProcessing class '{cls}' ({len(image_paths)} images found)...")
        if len(image_paths) < 5:
            print(f"  Warning: Class '{cls}' has fewer than 5 images. Accuracy may be lower.")
        
        for img_path in tqdm(image_paths):
            try:
                # Load image using PIL to ensure proper RGB
                pil_img = Image.open(img_path).convert("RGB")
                
                # Check quality (Blur)
                if is_blurry(pil_img, threshold=50.0): # Lower threshold for natural datasets
                    # print(f"Skipping {os.path.basename(img_path)} - too blurry")
                    rejected_images += 1
                    continue
                
                # If we want to detect the object first and crop it:
                if use_detector:
                    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    detections = detector.detect_and_crop(img_cv)
                    
                    if not detections:
                        rejected_images += 1
                        continue
                    
                    # Assume largest bounding box is the object of interest
                    detections.sort(key=lambda d: (d['bbox'][2]-d['bbox'][0]) * (d['bbox'][3]-d['bbox'][1]), reverse=True)
                    target_crop = detections[0]['crop']
                    
                    # Ensure valid crop
                    if target_crop.size == 0:
                        rejected_images += 1
                        continue
                    
                    embedding = embedder.get_embedding(target_crop)
                else:
                    # Treat the whole image as the object
                    embedding = embedder.get_embedding(pil_img)
                
                # Add to DB
                database.add_entry(embedding, cls)
                total_images_processed += 1
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                rejected_images += 1

    print(f"\nDatabase building complete!")
    print(f"Total objects added: {total_images_processed}")
    print(f"Images rejected (blurry/empty): {rejected_images}")
    print(f"Database saved to {db_dir}/")

if __name__ == "__main__":
    # The system now forces YOLO object detection to crop bounding boxes before DINOv2 embedding
    build_database(dataset_dir="dataset", db_dir="data", use_detector=True)
