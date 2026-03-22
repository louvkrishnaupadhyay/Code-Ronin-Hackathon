import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
import cv2

class Embedder:
    def __init__(self, model_name="facebook/dinov2-base"):
        """Initializes DINOv2 feature extractor without Meta tensors."""
        # 7. DINOv2 SAFETY: Load model WITHOUT meta/lazy settings.
        # DO NOT use device_map="auto".
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} on {self.device}...")

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        # 1. ROOT CAUSE FIX: Load with all weights eagerly.
        # 2. DO NOT use low_cpu_mem_usage or lazy map.
        self.model = AutoModel.from_pretrained(model_name, low_cpu_mem_usage=False)
        self.model.to(self.device)
        self.model.eval()

    def get_embedding(self, image):
        """Converts image to embedding."""
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            
        embeddings = outputs.last_hidden_state[:, 0]
        vector = embeddings[0].cpu().numpy()
        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            vector /= norm
        else:
            vector = np.zeros_like(vector)
        return vector
