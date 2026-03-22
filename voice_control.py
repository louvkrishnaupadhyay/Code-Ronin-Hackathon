"""
Hands-free voice commands. PyPI package name is SpeechRecognition (import name speech_recognition).

Install (same environment you use for Streamlit):
    python -m pip install SpeechRecognition
Optional for microphone input on Windows (often required):
    python -m pip install pyaudio
"""

import threading
import time

try:
    import speech_recognition as sr
except ImportError:
    sr = None

_MISSING_PKG = (
    "speech_recognition missing — install with: python -m pip install SpeechRecognition"
)


class VoiceController:
    """Background speech-to-text for hands-free commands. Degrades gracefully without mic or package."""

    def __init__(self, voice_assist):
        self.voice = voice_assist
        self.command_queue = []
        self.running = True
        self.detection_active = False
        self.available = False
        self._mic_error = None
        self.recognizer = None

        if sr is None:
            self._mic_error = _MISSING_PKG
            print(f"VoiceController: {_MISSING_PKG}")
            self.thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.thread.start()
            return

        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True

        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            self.available = True
        except Exception as e:
            self._mic_error = str(e)
            print(f"VoiceController: microphone unavailable ({e}). Voice commands disabled.")

        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        if self.available:
            print("Voice Controller initialized and listening...")

    def _listen_loop(self):
        while self.running:
            if sr is None or self.recognizer is None or not self.available:
                time.sleep(2)
                continue
            try:
                with sr.Microphone() as source:
                    audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=3)
                command = self.recognizer.recognize_google(audio).lower()
                print(f"User said: {command}")

                if "start detection" in command:
                    self.detection_active = True
                    self.voice.speak("Detection mode activated", priority=True)
                elif "stop" in command:
                    self.detection_active = False
                    self.voice.speak("Detection paused", priority=True)
                elif "describe surroundings" in command or "look" in command:
                    self.command_queue.append("describe")

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"Voice recognition service error: {e}")
                time.sleep(2)
            except Exception as e:
                print(f"Voice Recognition Error: {e}")
                time.sleep(1)

    def get_and_clear_command(self):
        if self.command_queue:
            return self.command_queue.pop(0)
        return None

    def stop(self):
        self.running = False
