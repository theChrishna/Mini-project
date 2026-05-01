import threading
import config

class VoiceOutput:
    def __init__(self):
        self.engine = None
        self.elevenlabs_client = None
        self.mode = "none"
        self._init_engine()
        self._last_spoken = ""

    def _init_engine(self):
        """Try ElevenLabs first, then pyttsx3, then silent mode."""
        # Attempt ElevenLabs
        if config.ELEVENLABS_API_KEY:
            try:
                from elevenlabs.client import ElevenLabs
                self.elevenlabs_client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
                self.mode = "elevenlabs"
                print("[Voice] Using ElevenLabs TTS.")
                return
            except Exception as e:
                print(f"[Voice] ElevenLabs init failed: {e}")

        # Fallback to pyttsx3
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 160)
            self.mode = "pyttsx3"
            print("[Voice] Using pyttsx3 (offline TTS).")
        except Exception as e:
            print(f"[Voice] pyttsx3 init failed: {e}. Voice output disabled.")
            self.mode = "none"

    def _emotion_to_voice_params(self, emotion):
        """Map detected emotion to ElevenLabs voice settings."""
        params = {
            "Happy":       {"stability": 0.3, "similarity_boost": 0.8, "style": 0.7},
            "Sad":         {"stability": 0.7, "similarity_boost": 0.6, "style": 0.2},
            "Angry":       {"stability": 0.2, "similarity_boost": 0.9, "style": 0.9},
            "Neutral":     {"stability": 0.5, "similarity_boost": 0.75, "style": 0.5},
            "Questioning": {"stability": 0.4, "similarity_boost": 0.7, "style": 0.6},
            "Skeptical":   {"stability": 0.6, "similarity_boost": 0.65, "style": 0.4},
        }
        return params.get(emotion, params["Neutral"])

    def _emotion_to_pyttsx3_rate(self, emotion):
        rates = {
            "Happy": 180,
            "Sad": 130,
            "Angry": 200,
            "Neutral": 160,
            "Questioning": 165,
            "Skeptical": 150,
        }
        return rates.get(emotion, 160)

    def speak(self, text, emotion="Neutral"):
        if not config.ENABLE_VOICE_OUTPUT or not text or text == self._last_spoken:
            return
        self._last_spoken = text
        threading.Thread(target=self._speak_async, args=(text, emotion), daemon=True).start()

    def _speak_async(self, text, emotion):
        if self.mode == "elevenlabs":
            self._speak_elevenlabs(text, emotion)
        elif self.mode == "pyttsx3":
            self._speak_pyttsx3(text, emotion)
        else:
            print(f"[Voice] (silent) {text}")

    def _speak_elevenlabs(self, text, emotion):
        try:
            import io
            params = self._emotion_to_voice_params(emotion)
            audio = self.elevenlabs_client.text_to_speech.convert(
                voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice
                text=text,
                model_id="eleven_monolingual_v1",
                voice_settings=params
            )
            import pygame
            pygame.mixer.init()
            audio_bytes = b"".join(audio)
            sound = pygame.mixer.Sound(io.BytesIO(audio_bytes))
            sound.play()
        except Exception as e:
            print(f"[Voice] ElevenLabs speech error: {e}")
            self._speak_pyttsx3(text, emotion)

    def _speak_pyttsx3(self, text, emotion):
        try:
            self.engine.setProperty("rate", self._emotion_to_pyttsx3_rate(emotion))
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"[Voice] pyttsx3 speech error: {e}")
