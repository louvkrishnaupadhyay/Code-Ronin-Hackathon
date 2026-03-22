import cv2
import PIL.Image
from detect import Detector
from embed import Embedder
from database import Database

class Learner:
    def __init__(self, detector: Detector, embedder: Embedder, database: Database):
        """Initializes the Learning module."""
        self.detector = detector
        self.embedder = embedder
        self.database = database

    def learn_new_object(self, image_input, label_name):
        """
        Learns a new object from an image input (either a path or a PIL image/Numpy array).
        Processes the image, detects (optional, or just uses the whole image as the object),
        extracts embedding, and stores in the database with the provided label.
        """
        # If it's a PIL/Numpy input, convert to numpy for detector and embedder if needed
        # Assuming image_input is already loaded properly.
        
        # In the context of learning, usually the user uploads a single image for one object.
        # We can either detect or use the entire image.
        # The prompt says: upload image, enter label.
        
        # Generate embedding
        embedding = self.embedder.get_embedding(image_input)
        
        # Store in FAISS
        self.database.add_entry(embedding, label_name)
        
        return True

    def learn_from_crop(self, crop, label_name):
        """Specifically learns from a cropped image snippet."""
        embedding = self.embedder.get_embedding(crop)
        self.database.add_entry(embedding, label_name)
        return True

    def learn_from_embedding(self, embedding, label_name):
        """Adds a pre-calculated embedding directly to the database."""
        self.database.add_entry(embedding, label_name)
        return True
