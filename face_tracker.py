import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import config
import os
from collections import deque


class FaceTracker:
    """
    Face tracking + emotion detection using MediaPipe FaceLandmarker.

    Emotion is derived from 52 pre-computed blendshape coefficients
    (already produced by FaceLandmarker at zero extra cost).
    Temporal smoothing with a rolling mode over the last N frames
    prevents flickering.
    """

    # Blendshape keys we care about
    _BS = {
        "smile_l":      "mouthSmileLeft",
        "smile_r":      "mouthSmileRight",
        "frown_l":      "mouthFrownLeft",
        "frown_r":      "mouthFrownRight",
        "brow_dn_l":    "browDownLeft",
        "brow_dn_r":    "browDownRight",
        "brow_in_up":   "browInnerUp",
        "brow_out_l":   "browOuterUpLeft",
        "brow_out_r":   "browOuterUpRight",
        "eye_wide_l":   "eyeWideLeft",
        "eye_wide_r":   "eyeWideRight",
        "eye_sq_l":     "eyeSquintLeft",
        "eye_sq_r":     "eyeSquintRight",
        "cheek_sq_l":   "cheekSquintLeft",
        "cheek_sq_r":   "cheekSquintRight",
        "jaw_open":     "jawOpen",
        "sneer_l":      "noseSneerLeft",
        "sneer_r":      "noseSneerRight",
    }

    SMOOTH_WINDOW = 12   # frames to smooth over (mode of last N detections)

    def __init__(self):
        model_path = os.path.join("models", "face_landmarker.task")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,           # ← key: enables blendshapes
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_face_presence_confidence=config.MIN_TRACKING_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector    = vision.FaceLandmarker.create_from_options(options)
        self._history    = deque(maxlen=self.SMOOTH_WINDOW)  # smoothing buffer
        self._last       = "Neutral"

    # ── MediaPipe interface ─────────────────────────────────────────────

    def process_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.detector.detect(mp_image)

    def draw_landmarks(self, frame, results):
        if results.face_landmarks:
            h, w = frame.shape[:2]
            for face_lms in results.face_landmarks:
                for lm in face_lms[::8]:   # sparse mesh — every 8th point
                    cv2.circle(frame,
                               (int(lm.x * w), int(lm.y * h)),
                               1, (200, 200, 200), -1)
        return frame

    # ── Emotion detection ───────────────────────────────────────────────

    def detect_emotion(self, results, frame=None):
        """
        Returns a smoothed emotion string using blendshape coefficients.
        'frame' parameter accepted but unused (kept for API compatibility).
        """
        if not results.face_blendshapes:
            self._history.append(self._last)
            return self._mode()

        # Build blendshape lookup: name → score (0..1)
        raw = {b.category_name: b.score for b in results.face_blendshapes[0]}
        bs  = {k: raw.get(v, 0.0) for k, v in self._BS.items()}

        emotion = self._classify(bs)
        self._history.append(emotion)
        self._last = self._mode()
        return self._last

    def _classify(self, bs):
        """
        Rule-based classifier on blendshape scores.
        Scores are 0.0 (not present) → 1.0 (maximum).
        """
        smile       = (bs["smile_l"]    + bs["smile_r"])    / 2
        cheek_sq    = (bs["cheek_sq_l"] + bs["cheek_sq_r"]) / 2
        frown       = (bs["frown_l"]    + bs["frown_r"])    / 2
        brow_dn     = (bs["brow_dn_l"]  + bs["brow_dn_r"])  / 2
        brow_in_up  = bs["brow_in_up"]
        brow_out    = (bs["brow_out_l"] + bs["brow_out_r"]) / 2
        eye_wide    = (bs["eye_wide_l"] + bs["eye_wide_r"]) / 2
        eye_sq      = (bs["eye_sq_l"]   + bs["eye_sq_r"])   / 2
        jaw_open    = bs["jaw_open"]
        sneer       = (bs["sneer_l"]    + bs["sneer_r"])    / 2

        # Happy: clear smile or cheek squint (genuine smile lifts cheeks)
        if smile > 0.40 or cheek_sq > 0.35:
            return "Happy"

        # Sad: frown + inner brow raised (the "worried" brow + downturned mouth)
        if frown > 0.20 and brow_in_up > 0.20:
            return "Sad"

        # Angry: brows pulled down AND eyes narrowed (squinting)
        if brow_dn > 0.30 and eye_sq > 0.25:
            return "Angry"

        # Questioning / Surprise: brows raised high OR eyes wide open
        if brow_out > 0.25 or eye_wide > 0.25 or (brow_in_up > 0.30 and jaw_open > 0.15):
            return "Questioning"

        # Skeptical: nose sneer (one-sided raised lip / disgust micro-expression)
        if sneer > 0.20:
            return "Skeptical"

        return "Neutral"

    def _mode(self):
        """Returns the most common emotion in the smoothing window."""
        if not self._history:
            return "Neutral"
        return max(set(self._history), key=self._history.count)

    # ── Body zone reference ─────────────────────────────────────────────

    def get_face_zones(self, results):
        """Y-coordinate body reference zones for position-aware sign recognition."""
        if not results.face_landmarks:
            return None
        lm          = results.face_landmarks[0]
        forehead_y  = lm[10].y
        chin_y      = lm[152].y
        face_height = chin_y - forehead_y
        return {
            "forehead_y":  forehead_y,
            "chin_y":      chin_y,
            "chest_y_est": chin_y + face_height * 0.65,
        }
