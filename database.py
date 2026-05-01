import firebase_admin
from firebase_admin import credentials, firestore
import os
import config

db = None

def init_db():
    global db
    if db is not None:
        return db

    if not os.path.exists(config.FIREBASE_CREDENTIALS_PATH):
        print(f"[WARNING] Firebase credentials not found at {config.FIREBASE_CREDENTIALS_PATH}. Database features will be mocked locally.")
        return None

    try:
        cred = credentials.Certificate(config.FIREBASE_CREDENTIALS_PATH)
        # Prevent initializing multiple times in Flask dev environment
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("[INFO] Connected to Firebase Firestore successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Firebase: {e}")
        db = None
    return db


# ---------------------------------------------------------
# Gestures
# ---------------------------------------------------------

def save_gesture(user_id, word, states, zone, motion):
    if not db: 
        print(f"[DB-MOCK] Saved gesture '{word}' to local memory.")
        return False
    try:
        doc_ref = db.collection('gestures').document()
        doc_ref.set({
            'word': word,
            'states': states,
            'zone': zone,
            'motion': motion,
            'created_at': firestore.SERVER_TIMESTAMP,
            'created_by': user_id
        })
        return True
    except Exception as e:
        print(f"[DB ERROR] Save gesture failed: {e}")
        return False

def load_gestures(user_id):
    """Load gestures created by the specific user."""
    if not db: return {}
    try:
        gestures = {}
        # Fetch user's own gestures
        # NOTE: Firebase requires a composite index if we do complex querying, but simple equality is fine.
        docs = db.collection('gestures').where('created_by', '==', user_id).stream()
        for doc in docs:
            data = doc.to_dict()
            gestures[data['word']] = data
        
        return gestures
    except Exception as e:
        print(f"[DB ERROR] Load gestures failed: {e}")
        return {}


# ---------------------------------------------------------
# History Logging
# ---------------------------------------------------------

def log_history(user_id, raw_buffer, translated_text, emotion):
    if not db: 
        print(f"[DB-MOCK] Logged history: {translated_text}")
        return False
    try:
        doc_ref = db.collection('history').document()
        doc_ref.set({
            'user_id': user_id,
            'timestamp': firestore.SERVER_TIMESTAMP,
            'raw_buffer': raw_buffer,
            'translated_text': translated_text,
            'emotion': emotion
        })
        return True
    except Exception as e:
        print(f"[DB ERROR] Log history failed: {e}")
        return False


# ---------------------------------------------------------
# User Settings
# ---------------------------------------------------------

def save_settings(user_id, toggles):
    if not db: return False
    try:
        doc_ref = db.collection('settings').document(user_id)
        doc_ref.set(toggles, merge=True)
        return True
    except Exception as e:
        print(f"[DB ERROR] Save settings failed: {e}")
        return False

def load_settings(user_id):
    if not db: return None
    try:
        doc_ref = db.collection('settings').document(user_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
    except Exception as e:
        print(f"[DB ERROR] Load settings failed: {e}")
    return None

# Auto-initialize
init_db()
