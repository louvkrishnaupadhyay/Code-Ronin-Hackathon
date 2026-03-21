import cv2
import time
from src.detector import ObjectDetector
from src.embedder import FeatureEmbedder
from src.search import VectorSearchEngine
from src.voice import VoiceInterface

def main():
    # ---------------------------
    # Initialization
    # ---------------------------
    print("Initializing CodeNova Adaptive Vision system...")
    
    # 1. Detection (YOLOv8)
    detector = ObjectDetector(model_name='yolov8n.pt')
    
    # 2. Embedding (DINOv2)
    # Note: VITS14 gives 384-dim embeddings
    embedder = FeatureEmbedder(model_name='dinov2_vits14')
    
    # 3. Search Engine (FAISS)
    search_engine = VectorSearchEngine(embedding_dim=384)
    
    # 4. Voice Interaction (SpeechRecognition + pyttsx3)
    voice = VoiceInterface()
    
    # Start Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("System active! Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # STEP 1: DETECTION
        detections = detector.detect_objects(frame)
        
        for det in detections:
            bbox = det['bbox']
            
            # STEP 2: CROPPING & FEATURE EXTRACTION
            crop = detector.crop_and_extract(frame, bbox)
            if crop.size == 0: continue
            
            embedding = embedder.get_embedding(crop)
            
            # STEP 3: SIMILARITY SEARCH
            label, score = search_engine.search_object(embedding, threshold=0.6)
            
            # STEP 4: ACTION
            color = (0, 255, 0) # Green for known
            
            if label == "Unknown":
                color = (0, 0, 255) # Red for unknown
                
                # Show frame with the highlighted object so the user can see it during the voice interaction
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(frame, "STATUS: UNKNOWN", (bbox[0], bbox[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.imshow("CodeNova Preview", frame)
                cv2.waitKey(100) # Quick refresh
                
                # VOICE INTERACTION: Ask for a label
                new_label = voice.listen_for_label()
                if new_label:
                    search_engine.add_object(embedding, new_label)
                    label = new_label
                    color = (0, 255, 0) # Update to green
            
            # Draw persistent info
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, f"{label} ({score:.2f})", (bbox[0], bbox[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Show Output
        cv2.imshow("CodeNova Preview", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("CodeNova deactivated.")

if __name__ == "__main__":
    main()
