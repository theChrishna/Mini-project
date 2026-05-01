# Configuration for ASL Voice Translator

# System Toggles
ENABLE_HAND_TRACKING = True
ENABLE_EMOTION_DETECTION = True
ENABLE_LLM_TRANSLATION = True # Enabled for Gemini ASL-to-English translation
ENABLE_VOICE_OUTPUT = True

# --- Database & Authentication ---
FIREBASE_CREDENTIALS_PATH = "firebase-credentials.json"
CURRENT_USER_ID           = "demo_user_1"  # Replace with real auth later

# Camera
CAMERA_INDEX   = 2      # 0 = built-in | 2 = DroidCam (detected)
CAMERA_BACKEND = "DSHOW" # "DSHOW" for Windows (needed for DroidCam) | "" for default

# Tracking Parameters
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5
TARGET_FPS = 30

# API Keys and Models
ELEVENLABS_API_KEY = "" # Ensure this remains empty until provided
GEMINI_API_KEY = "AIzaSyDo3e1XJTReLakQfCbBIneQzt-RMHIjZXw" # Enter your Gemini API key here
OLLAMA_MODEL = "llama3"

# Sign Recognition Thresholds
SIGN_HOLD_SECONDS      = 1.0  # Seconds a sign must be held to register
SIGN_COOLDOWN_SECONDS  = 1.0  # Seconds to block after a sign registers
AUTO_SPEAK_ON_SILENCE  = True
AUTO_SPEAK_SILENCE_SECS = 2.5

# Emotion Categories
EMOTIONS = ["Happy", "Sad", "Angry", "Neutral", "Questioning", "Skeptical"]
