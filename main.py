"""
ASL Voice Translator — Main Entry Point
========================================
Controls the main event loop:
  - OpenCV webcam capture
  - MediaPipe hand & face tracking
  - Rule-based ASL sign recognition
  - LLM interpretation (Ollama, with bypass if unavailable)
  - Text-to-Speech output (ElevenLabs or pyttsx3 fallback)

Keyboard Controls:
  H — Toggle hand tracking
  E — Toggle emotion detection
  L — Toggle LLM interpretation
  V — Toggle voice output
  C — Clear sign buffer
  S — Speak current translated sentence
  Q — Quit
"""

import time
import cv2
import config

from hand_tracker   import HandTracker
from face_tracker   import FaceTracker
from sign_recognizer import SignRecognizer
from llm_interpreter import LLMInterpreter
from voice_output   import VoiceOutput
from ui             import draw_ui_overlay


def main():
    backend = cv2.CAP_DSHOW if config.CAMERA_BACKEND == "DSHOW" else cv2.CAP_ANY
    cap = cv2.VideoCapture(config.CAMERA_INDEX, backend)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

    # --- Initialise modules ---
    print("[INFO] Initializing HandTracker...")
    hand_tracker   = HandTracker()
    print("[INFO] Initializing FaceTracker...")
    face_tracker   = FaceTracker()
    print("[INFO] Initializing SignRecognizer...")
    sign_recognizer = SignRecognizer()
    print("[INFO] Initializing LLMInterpreter...")
    llm            = LLMInterpreter()
    print("[INFO] Initializing VoiceOutput...")
    voice          = VoiceOutput()

    # --- Mutable toggle state ---
    toggles = {
        "hand":    config.ENABLE_HAND_TRACKING,
        "emotion": config.ENABLE_EMOTION_DETECTION,
        "llm":     config.ENABLE_LLM_TRANSLATION,
        "voice":   config.ENABLE_VOICE_OUTPUT,
    }

    # --- Runtime state ---
    emotion              = "Neutral"
    translated_text      = ""
    prev_time            = time.time()
    last_sign_time       = time.time()   # tracks when last sign was added
    auto_spoken          = False          # prevents repeated auto-speaks
    gesture              = None
    gesture_conf         = 0.0

    print("[INFO] ASL Voice Translator started. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read webcam frame.")
            break

        frame = cv2.flip(frame, 1)  # Mirror for natural feel

        # --- FPS calculation ---
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-6)
        prev_time = now

        # --- Hand tracking ---
        hand_landmarks_list = []
        gesture             = None
        gesture_conf        = 0.0
        if toggles["hand"]:
            hand_results        = hand_tracker.process_frame(frame)
            frame               = hand_tracker.draw_landmarks(frame, hand_results)
            hand_landmarks_list = hand_tracker.get_landmarks(hand_results)
            gesture, gesture_conf = hand_tracker.get_gesture_info(hand_results)

        # --- Face / emotion tracking ---
        face_zones = None
        if toggles["emotion"]:
            face_results = face_tracker.process_frame(frame)
            frame        = face_tracker.draw_landmarks(frame, face_results)
            emotion      = face_tracker.detect_emotion(face_results, frame)
            face_zones   = face_tracker.get_face_zones(face_results)

        # --- Sign recognition & sentence buffer ---
        sign_buffer  = sign_recognizer.update(hand_landmarks_list, gesture=gesture, face_zones=face_zones)
        current_sign = sign_recognizer.get_current_sign()

        # --- Manual Sentence Speaking (Toggle Stop) ---
        if sign_recognizer.just_toggled and not sign_recognizer.is_recording:
            buf = sign_recognizer.get_buffer()
            if buf and toggles["voice"]:
                sentence = llm.translate(" ".join(buf), emotion)
                translated_text = sentence
                voice.speak(sentence, emotion)
                print(f"[MANUAL-SPEAK] {sentence}")
            elif not buf:
                # Buffer was empty, just clear the UI text
                translated_text = ""

        # --- UI overlay ---
        frame = draw_ui_overlay(
            frame,
            sign_buffer=sign_buffer,
            current_sign=current_sign,
            emotion=emotion,
            translated_text=translated_text,
            toggles=toggles,
            fps=fps,
            hold_progress=sign_recognizer.get_hold_progress(),
            gesture_conf=gesture_conf,
            is_recording=sign_recognizer.is_recording,
        )

        cv2.imshow("ASL Voice Translator", frame)

        # --- Keyboard input ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            print("[INFO] Quitting.")
            break

        elif key == ord('h') or key == ord('H'):
            toggles["hand"] = not toggles["hand"]
            print(f"[TOGGLE] Hand Tracking: {'ON' if toggles['hand'] else 'OFF'}")

        elif key == ord('e') or key == ord('E'):
            toggles["emotion"] = not toggles["emotion"]
            print(f"[TOGGLE] Emotion Detection: {'ON' if toggles['emotion'] else 'OFF'}")

        elif key == ord('l') or key == ord('L'):
            toggles["llm"] = not toggles["llm"]
            config.ENABLE_LLM_TRANSLATION = toggles["llm"]
            print(f"[TOGGLE] LLM Translation: {'ON' if toggles['llm'] else 'OFF'}")

        elif key == ord('v') or key == ord('V'):
            toggles["voice"] = not toggles["voice"]
            config.ENABLE_VOICE_OUTPUT = toggles["voice"]
            print(f"[TOGGLE] Voice Output: {'ON' if toggles['voice'] else 'OFF'}")

        elif key == ord('c') or key == ord('C'):
            sign_recognizer.clear_buffer()
            translated_text = ""
            auto_spoken     = False
            print("[INFO] Buffer cleared.")

        elif key == ord('s') or key == ord('S'):
            # Translate the current buffer and speak
            buffer = sign_recognizer.get_buffer()
            if buffer:
                sentence = llm.translate(" ".join(buffer), emotion)
                translated_text = sentence
                if toggles["voice"]:
                    voice.speak(sentence, emotion)
                print(f"[OUTPUT] {sentence}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
