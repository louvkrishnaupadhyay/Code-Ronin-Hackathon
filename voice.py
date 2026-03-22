import pyttsx3
import threading
import queue
import time

class VoiceAssist:
    def __init__(self):
        """Initializes the Text-to-Speech system with a background worker thread."""
        self.msg_queue = queue.Queue()
        self.last_announced = {}  # Label -> Last announcement time
        self.cooldown = 2.0  # Seconds between repeated announcements for the same object/position
        self.last_summary = ""
        self.last_summary_time = 0
        self.speech_enabled = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        print("🎙️ Non-blocking Voice Engine Initialized")

    def set_speech_enabled(self, enabled: bool):
        """When False, speak() and blind-mode announcements are suppressed."""
        self.speech_enabled = bool(enabled)

    def _worker(self):
        """Dedicated background thread for sequential speech processing."""
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass
        engine = None
        while True:
            try:
                if engine is None:
                    engine = pyttsx3.init()
                    engine.setProperty('rate', 160)
                    engine.setProperty('volume', 1.0)
                
                text = self.msg_queue.get(timeout=1)
                if text:
                    print(f"AI SPEAKING: {text}")
                    # Re-init check before speaking to avoid stale engine
                    try:
                        engine.say(text)
                        engine.runAndWait()
                    except Exception as e:
                        print(f"Speech Exception: {e}")
                        engine = None # Force re-init next time
                self.msg_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Voice Worker Loop Error: {e}")
                engine = None
                time.sleep(1)

    def speak(self, text, priority=False):
        """Adds text to the background speech queue."""
        if not self.speech_enabled:
            return
        if text:
            if priority:
                while not self.msg_queue.empty():
                    try: self.msg_queue.get_nowait()
                    except: break
            self.msg_queue.put(text)

    def _get_direction(self, bbox, frame_width):
        """Calculates object direction based on bbox center."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        
        if center_x < frame_width * 0.3:
            return "on your left"
        elif center_x > frame_width * 0.7:
            return "on your right"
        return "in front of you"

    def _get_distance(self, bbox, frame_area):
        """Estimates distance based on relative bbox size."""
        x1, y1, x2, y2 = bbox
        obj_area = (x2 - x1) * (y2 - y1)
        ratio = obj_area / frame_area
        
        if ratio > 0.35:
            return "very close"
        elif ratio > 0.08:
            return "nearby"
        return "at a distance"

    def _get_smart_description(self, label, direction, distance):
        """Context-aware smart descriptions."""
        lbl = label.lower()
        if "car" in lbl or "vehicle" in lbl or "truck" in lbl or "bus" in lbl:
            return f"Warning: vehicle {distance} {direction}"
        return None  # Return None if no special warning, so it can be grouped.

    def _announce_cooldown_key(self, det, frame_width):
        """Same class in different directions can be announced separately."""
        lab = (det.get("announce_label") or det.get("label") or "object").lower()
        direction = self._get_direction(det["bbox"], frame_width)
        return f"{lab}|{direction}"

    def notify_blind_mode(self, detections, frame_width, frame_height):
        """Intelligent auditory feedback for blind users."""
        if not self.speech_enabled or not detections:
            return

        frame_area = frame_width * frame_height
        # Use vision (YOLO) names when memory does not match — caller sets announce_label / label
        valid_dets = []
        for d in detections:
            ann = d.get("announce_label") or d.get("label")
            if ann and str(ann).strip() and str(ann).lower() != "unknown":
                valid_dets.append(d)
        if not valid_dets:
            return

        now = time.time()
        
        warnings = []
        dir_map = {}
        
        for det in valid_dets:
            label = (det.get("announce_label") or det.get("label") or "object").lower()
            ck = self._announce_cooldown_key(det, frame_width)
            if now - self.last_announced.get(ck, 0) <= self.cooldown:
                continue

            direction = self._get_direction(det["bbox"], frame_width)
            distance = self._get_distance(det["bbox"], frame_area)
            
            smart_desc = self._get_smart_description(label, direction, distance)
            if smart_desc:
                warnings.append(smart_desc)
            else:
                if direction not in dir_map:
                    dir_map[direction] = []
                dir_map[direction].append(label)
                
            self.last_announced[ck] = now

        to_announce = warnings.copy()
        
        for direction, labels in dir_map.items():
            unique_labels = list(set(labels))
            if len(unique_labels) == 1:
                to_announce.append(f"A {unique_labels[0]} is {direction}")
            else:
                last = unique_labels.pop()
                labels_str = ", ".join([f"a {l}" for l in unique_labels]) + f" and a {last}"
                to_announce.append(f"There is {labels_str} {direction}")

        if to_announce:
            summary = ". ".join(to_announce) + "."
            if summary != self.last_summary or (now - getattr(self, 'last_summary_time', 0) > 10):
                self.speak(summary)
                self.last_summary = summary
                self.last_summary_time = now

    def generate_scene_description(self, detections):
        """Describe the scene using memory names when present, otherwise vision model class names."""
        seen = []
        for d in detections:
            lab = d.get("announce_label") or d.get("label")
            if not lab or str(lab).strip().lower() == "unknown":
                continue
            low = lab.lower()
            if low not in seen:
                seen.append(low)
        if not seen:
            return "I don't see any objects in view right now. Try moving the camera slowly."

        intro = "I see"
        if len(seen) == 1:
            return f"{intro} a {seen[0]}."
        if len(seen) == 2:
            return f"{intro} a {seen[0]} and a {seen[1]}."
        last = seen[-1]
        rest = ", ".join(seen[:-1])
        return f"{intro} {rest}, and a {last}."

    def notify_detection(self, labels):
        """Legacy support for normal UI mode."""
        if not labels: return
        unique_labels = list(set(labels))
        text = f"Spotted: {', '.join(unique_labels)}"
        self.speak(text)
