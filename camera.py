import time
import cv2
import config

from hand_tracker   import HandTracker
from face_tracker   import FaceTracker
from sign_recognizer import SignRecognizer
from llm_interpreter import LLMInterpreter
from voice_output   import VoiceOutput
from ui             import draw_ui_overlay
import database

class VideoCamera:
    """
    Encapsulates the OpenCV webcam and MediaPipe loop so it can be served over HTTP via Flask.
    """
    def __init__(self):
        backend = cv2.CAP_DSHOW if config.CAMERA_BACKEND == "DSHOW" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX, backend)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

        self.hand_tracker   = HandTracker()
        self.face_tracker   = FaceTracker()
        self.sign_recognizer = SignRecognizer()
        self.llm            = LLMInterpreter()
        self.voice          = VoiceOutput()

        db_settings = database.load_settings(config.CURRENT_USER_ID)
        if db_settings:
            self.toggles = db_settings
            # Sync global config
            config.ENABLE_HAND_TRACKING = self.toggles.get("hand", config.ENABLE_HAND_TRACKING)
            config.ENABLE_EMOTION_DETECTION = self.toggles.get("emotion", config.ENABLE_EMOTION_DETECTION)
            config.ENABLE_LLM_TRANSLATION = self.toggles.get("llm", config.ENABLE_LLM_TRANSLATION)
            config.ENABLE_VOICE_OUTPUT = self.toggles.get("voice", config.ENABLE_VOICE_OUTPUT)
        else:
            self.toggles = {
                "hand":    config.ENABLE_HAND_TRACKING,
                "emotion": config.ENABLE_EMOTION_DETECTION,
                "llm":     config.ENABLE_LLM_TRANSLATION,
                "voice":   config.ENABLE_VOICE_OUTPUT,
            }

        self.emotion              = "Neutral"
        self.translated_text      = ""
        self.prev_time            = time.time()
        self.gesture_conf         = 0.0

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)

        # FPS calculation
        now = time.time()
        fps = 1.0 / (now - self.prev_time + 1e-6)
        self.prev_time = now

        # Hand tracking
        hand_landmarks_list = []
        gesture             = None
        self.gesture_conf   = 0.0
        if self.toggles["hand"]:
            hand_results        = self.hand_tracker.process_frame(frame)
            frame               = self.hand_tracker.draw_landmarks(frame, hand_results)
            hand_landmarks_list = self.hand_tracker.get_landmarks(hand_results)
            gesture, self.gesture_conf = self.hand_tracker.get_gesture_info(hand_results)

        # Face tracking
        face_zones = None
        if self.toggles["emotion"]:
            face_results = self.face_tracker.process_frame(frame)
            frame        = self.face_tracker.draw_landmarks(frame, face_results)
            self.emotion = self.face_tracker.detect_emotion(face_results, frame)
            face_zones   = self.face_tracker.get_face_zones(face_results)

        # Sign recognition
        sign_buffer  = self.sign_recognizer.update(hand_landmarks_list, gesture=gesture, face_zones=face_zones)
        current_sign = self.sign_recognizer.get_current_sign()

        # Manual Speak (Triggered by dropping toggle gesture)
        if self.sign_recognizer.just_toggled and not self.sign_recognizer.is_recording:
            buf = self.sign_recognizer.get_buffer()
            if buf and self.toggles["voice"]:
                raw_buf_str = " ".join(buf)
                sentence = self.llm.translate(raw_buf_str, self.emotion)
                self.translated_text = sentence
                self.voice.speak(sentence, self.emotion)
                database.log_history(config.CURRENT_USER_ID, raw_buf_str, sentence, self.emotion)
            elif not buf:
                self.translated_text = ""

        # UI Overlay
        frame = draw_ui_overlay(
            frame,
            sign_buffer=sign_buffer,
            current_sign=current_sign,
            emotion=self.emotion,
            translated_text=self.translated_text,
            toggles=self.toggles,
            fps=fps,
            hold_progress=self.sign_recognizer.get_hold_progress(),
            gesture_conf=self.gesture_conf,
            is_recording=self.sign_recognizer.is_recording,
        )

        # Encode frame to JPEG for web streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            return None
        return jpeg.tobytes()

    # --- API Methods for Web Buttons ---
    def toggle_feature(self, feature):
        if feature in self.toggles:
            self.toggles[feature] = not self.toggles[feature]
            if feature == "llm": config.ENABLE_LLM_TRANSLATION = self.toggles["llm"]
            if feature == "voice": config.ENABLE_VOICE_OUTPUT = self.toggles["voice"]
            database.save_settings(config.CURRENT_USER_ID, self.toggles)
        return self.toggles

    def clear_buffer(self):
        self.sign_recognizer.clear_buffer()
        self.translated_text = ""
        
    def start_custom_recording(self, word):
        self.sign_recognizer.start_custom_recording(word)
        
    def speak_manual(self):
        buffer = self.sign_recognizer.get_buffer()
        if buffer:
            raw_buf_str = " ".join(buffer)
            sentence = self.llm.translate(raw_buf_str, self.emotion)
            self.translated_text = sentence
            if self.toggles["voice"]:
                self.voice.speak(sentence, self.emotion)
            database.log_history(config.CURRENT_USER_ID, raw_buf_str, sentence, self.emotion)
