import pyttsx3
import threading
import queue
import time

class VoiceAssist:
    def __init__(self):
        """Initializes the Text-to-Speech system with a background worker thread."""
        self.msg_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        print("🎙️ Non-blocking Voice Engine Initialized")

    def _worker(self):
        """Dedicated background thread for sequential speech processing."""
        # Initialize the engine inside the thread that will use it
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
        except Exception as e:
            print(f"Failed to init voice engine: {e}")
            return

        while True:
            try:
                # Wait for next text message
                text = self.msg_queue.get(timeout=1)
                if text:
                    print(f"AI Speaking: {text}")
                    engine.say(text)
                    engine.runAndWait()
                self.msg_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                # If engine crashes, try to recover
                try: engine = pyttsx3.init()
                except: break

    def speak(self, text):
        """Adds text to the background speech queue. Returns immediately."""
        if text:
            # Avoid repeating exactly the same message too fast
            self.msg_queue.put(text)

    def notify_detection(self, labels):
        """Provides a vocal summary of detected objects (non-blocking)."""
        if not labels:
            return
            
        unique_labels = list(set(labels))
        
        if len(unique_labels) == 1:
            text = f"Detected {unique_labels[0]}"
        else:
            text = f"Objects found: {', '.join(unique_labels)}"
            
        self.speak(text)
