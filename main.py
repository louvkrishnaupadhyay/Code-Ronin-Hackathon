import os
import cv2
import time
import numpy as np
from detect import Detector
from embed import Embedder
from database import Database, QUANTUM_MODE
from voice import VoiceAssist
from learn import Learner

try:
    import speech_recognition as sr
except ImportError:
    sr = None


def main():
    print("Initializing CodeNova Intelligent Assistive AI System...")
    
    # 1. Models (Using the verified root-level architecture)
    detector = Detector(model_path='yolov8s.pt')
    embedder = Embedder(model_name='facebook/dinov2-base')
    database = Database(embedding_dim=768)
    learner = Learner(detector, embedder, database)
    voice = VoiceAssist()
    
    recognizer = sr.Recognizer() if sr else None

    def listen_for_label():
        """Listen for the user to provide a label via voice."""
        if sr is None or recognizer is None:
            print("Voice labeling skipped — install SpeechRecognition: python -m pip install SpeechRecognition")
            return None
        with sr.Microphone() as source:
            voice.speak("I don't recognize this object. What should I call it?", priority=True)
            time.sleep(2) # Give Pyttsx3 a moment to announce
            recognizer.adjust_for_ambient_noise(source)
            try:
                print("🎙️ Listening...")
                audio = recognizer.listen(source, timeout=4, phrase_time_limit=3)
                label = recognizer.recognize_google(audio)
                new_label = label.strip()
                voice.speak(f"Okay, synchronizing memory with: {new_label}", priority=True)
                return new_label
            except sr.WaitTimeoutError:
                voice.speak("I didn't catch that. Returning to scanning.", priority=True)
                return None
            except Exception as e:
                voice.speak("Speech recognition failed. Skipping.", priority=True)
                return None

    # Windows: DirectShow is more reliable than MSMF for many USB webcams
    _root = os.path.dirname(os.path.abspath(__file__))
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(0)
    if not cap.isOpened() or not cap.read()[0]:
        print("Warning: Primary camera open failed. Retrying default backend...")
        cap.release()
        cap = cv2.VideoCapture(0)

    fallback_mode = False
    if not cap.isOpened() or not cap.read()[0]:
        print("Error: Could not open the webcam. Falling back to Demo Image Mode.")
        fallback_mode = True

    print("System active! Press 'q' to exit the CV viewport.")
    cv2.namedWindow("CodeNova AI Assist", cv2.WINDOW_NORMAL)
    
    fallback_image_path = os.path.join(_root, "data", "demo.jpg")
    
    while True:
        if fallback_mode:
            frame = cv2.imread(fallback_image_path)
            if frame is None:
                print("No data/demo.jpg — using synthetic demo frame.")
                h, w = 600, 800
                frame = np.zeros((h, w, 3), dtype=np.uint8)
                frame[:] = (45, 48, 55)
                cv2.putText(
                    frame, "CODENOVA DEMO", (w // 8, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 220, 80), 2, cv2.LINE_AA,
                )
            # Resize slightly if too large
            frame = cv2.resize(frame, (800, 600))
            time.sleep(0.1)  # Throttle mock frame rate
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Camera feed lost or unavailable. Switching to fallback mode.")
                fallback_mode = True
                continue
            
        # STEP 1: DETECTION & CROPPING
        detections = detector.detect_and_crop(frame)
        h, w, _ = frame.shape
        valid_dets = []
        
        for det in detections:
            bbox = det['bbox']
            crop = det['crop']
            if crop.size == 0: continue
            
            # STEP 2: DINOv2 EMBEDDING
            embedding = embedder.get_embedding(crop)
            
            # STEP 3: SIMILARITY SEARCH (Quantum Hybrid / FAISS Cosine)
            label, score = database.search_entry(embedding, threshold=0.50)
            
            det['label'], det['score'] = label, score
            color = (0, 255, 0) # Green for known
            
            if label == "Unknown":
                color = (0, 0, 255) # Red
                
                # Show frame with the highlighted object for User Feedback
                temp_frame = detector.draw_bbox(frame.copy(), bbox, "STATUS: UNKNOWN", color)
                cv2.imshow("CodeNova AI Assist", temp_frame)
                cv2.waitKey(1) # Refresh UI instantly
                
                # STEP 4: NO MATCH -> VOICE ASSIST -> STORE IN DB
                new_label = listen_for_label()
                if new_label:
                    learner.learn_from_embedding(embedding, new_label)
                    det['label'] = new_label
                    color = (0, 255, 0)
            else:
                valid_dets.append(det)
                
            frame = detector.draw_bbox(frame, bbox, f"{det['label']} ({score:.2f})", color)

        # STEP 5: MATCH FOUND -> BLIND MODE INTELLIGENT VOICE GUIDANCE
        if valid_dets:
            voice.notify_blind_mode(valid_dets, w, h)
            
        cv2.imshow("CodeNova AI Assist", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("CodeNova deactivated.")

if __name__ == "__main__":
    main()

