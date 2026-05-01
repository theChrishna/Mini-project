import config
import math
import time
import json
import os
import database
from collections import deque


class MotionBuffer:
    """Tracks wrist (x, y) over recent frames to classify motion type."""
    THRESHOLD = 0.035  # min movement range to count as non-static

    def __init__(self, maxlen=20):
        self.positions = deque(maxlen=maxlen)

    def add(self, wrist_xy):
        self.positions.append(wrist_xy)

    def get_motion(self):
        if len(self.positions) < 10:
            return "STATIC"
        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]
        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        if x_range < self.THRESHOLD and y_range < self.THRESHOLD:
            return "STATIC"
        if x_range > y_range * 1.5:
            return "LATERAL"
        if y_range > x_range * 1.5:
            return "VERTICAL"
        return "CIRCULAR"

    def reset(self):
        self.positions.clear()


class SignRecognizer:

    def __init__(self):
        self.sentence_buffer  = []
        self.last_sign        = None
        self._sign_start_time = None   # when the current sign was first seen
        self._cooldown_until  = 0.0    # epoch time until which new signs are blocked
        self.word_confirmed   = False
        self._prev_buffer_len = 0
        self.motion_buf       = MotionBuffer()

        # Recording state
        self.is_recording       = False
        self._toggle_start_time = None
        self._comma_start_time  = None
        self.just_toggled       = False

        # Custom Gesture Recording State
        self.custom_gestures = {}
        self._custom_record_word = None
        self._custom_record_start = None
        self._load_custom_gestures()

    def _load_custom_gestures(self):
        # Load from Firebase (or local fallback in db layer)
        self.custom_gestures = database.load_gestures(config.CURRENT_USER_ID)
        print(f"[INFO] Loaded {len(self.custom_gestures)} custom gestures from DB.")

    def _save_custom_gesture(self, word, states, zone, motion):
        # Save to Firebase DB
        database.save_gesture(config.CURRENT_USER_ID, word, list(states), zone, motion)
        # Update local dictionary
        self.custom_gestures[word] = {
            "states": list(states),
            "zone": zone,
            "motion": motion
        }
        print(f"[INFO] Saved custom gesture for '{word}' to DB!")

    def start_custom_recording(self, word):
        """Called by WebApp API to initiate recording of a new sign."""
        self._custom_record_word = word.upper()
        self._custom_record_start = None
        print(f"[INFO] Entering custom recording mode for: {self._custom_record_word}")

    # ── Progress helpers ────────────────────────────────────────────────

    def get_hold_progress(self):
        """0.0–1.0 fraction of hold completed for the current sign or toggle gesture."""
        if self._toggle_start_time is not None:
            return min((time.time() - self._toggle_start_time) / 1.0, 1.0)
        
        if self._sign_start_time is None or self.last_sign is None:
            return 0.0
        elapsed = time.time() - self._sign_start_time
        return min(elapsed / config.SIGN_HOLD_SECONDS, 1.0)

    def buffer_grew(self):
        grew = len(self.sentence_buffer) > self._prev_buffer_len
        self._prev_buffer_len = len(self.sentence_buffer)
        return grew

    # ── Geometry helpers ─────────────────────────────────────────────────

    def _get_finger_states(self, lm):
        if not lm:
            return None
        tips = [4, 8, 12, 16, 20]
        pips = [3, 6, 10, 14, 18]
        states = []
        if lm[0][0] < lm[tips[0]][0]:
            states.append(lm[tips[0]][0] > lm[pips[0]][0])
        else:
            states.append(lm[tips[0]][0] < lm[pips[0]][0])
        for i in range(1, 5):
            states.append(lm[tips[i]][1] < lm[pips[i]][1])
        return states

    def _dist(self, lm, a, b):
        return math.hypot(lm[a][0] - lm[b][0], lm[a][1] - lm[b][1])

    def _hand_size(self, lm):
        return self._dist(lm, 0, 12) + 1e-6

    def _curl(self, lm, tip, pip, mcp):
        tip_mcp = self._dist(lm, tip, mcp)
        pip_mcp = self._dist(lm, pip, mcp)
        return 1.0 - min(tip_mcp / (pip_mcp + 1e-6), 1.0)

    def _thumb_near(self, lm, finger_tip_idx, threshold=0.10):
        return self._dist(lm, 4, finger_tip_idx) / self._hand_size(lm) < threshold

    def _get_zone(self, wrist_y, face_zones):
        if face_zones is None:
            return "NEUTRAL"
        fy  = face_zones["forehead_y"]
        cy  = face_zones["chin_y"]
        chy = face_zones["chest_y_est"]
        if wrist_y < fy:                        return "ABOVE_HEAD"
        elif wrist_y < fy + (cy - fy) * 0.5:   return "FOREHEAD"
        elif wrist_y < cy:                       return "FACE"
        elif wrist_y < cy + (chy - cy) * 0.4:   return "CHIN"
        elif wrist_y < chy:                      return "CHEST"
        else:                                    return "BELLY"

    # ── Tier 2: Position + motion word signs ─────────────────────────────

    def _classify_word_sign(self, lm, states, zone, motion):
        if states is None:
            return None
        
        # --- Check Custom Gestures First ---
        for word, heuristics in self.custom_gestures.items():
            if list(states) == heuristics["states"] and zone == heuristics["zone"] and motion == heuristics["motion"]:
                return word

        thumb, index, middle, ring, pinky = states
        b_shape = all(states)
        fist    = not index and not middle and not ring and not pinky
        y_shape = thumb and not index and not middle and not ring and pinky

        # HELLO: flat hand at forehead, lateral sweep
        if b_shape and zone == "FOREHEAD" and motion == "LATERAL":   return "HELLO"
        # SICK: flat hand touching forehead, still
        if b_shape and zone == "FOREHEAD" and motion == "STATIC":    return "SICK"
        # GOODBYE: flat hand at face level, wave
        if b_shape and zone == "FACE" and motion == "LATERAL":       return "GOODBYE"
        # THANK YOU: flat hand at chin, moving down
        if b_shape and zone == "CHIN" and motion == "VERTICAL":      return "THANK YOU"
        # GOOD: flat hand at chin, still
        if b_shape and zone == "CHIN" and motion == "STATIC":        return "GOOD"
        # PLEASE: flat hand on chest, circular
        if b_shape and zone == "CHEST" and motion == "CIRCULAR":     return "PLEASE"
        # FINE: flat hand on chest, still
        if b_shape and zone == "CHEST" and motion == "STATIC":       return "FINE"
        # WAIT: flat hand at chest, side wiggle
        if b_shape and zone == "CHEST" and motion == "LATERAL":      return "WAIT"
        # SORRY: fist on chest, circular rub
        if fist and zone == "CHEST" and motion == "CIRCULAR":        return "SORRY"
        # MORE: fist at neutral, tapping (vertical)
        if fist and zone in ("NEUTRAL", "BELLY") and motion == "VERTICAL": return "MORE"
        # EAT / FOOD: bunched hand to mouth
        if fist and zone == "FACE" and motion == "STATIC":           return "EAT"
        # CALL / PHONE: Y-shape near ear
        if y_shape and zone == "FACE" and motion == "STATIC":        return "CALL"
        # DANGER: index+thumb (L-ish), lateral thrust
        if thumb and index and not middle and not ring and not pinky and motion == "LATERAL":
            return "DANGER"

        return None

    def _is_toggle_gesture(self, hand_landmarks_list):
        """Returns True if exactly 2 hands are detected, both fully open (flat)."""
        if not hand_landmarks_list or len(hand_landmarks_list) != 2:
            return False
        
        for lm in hand_landmarks_list:
            states = self._get_finger_states(lm)
            if not states or not all(states):
                return False
        return True

    def _is_comma_gesture(self, hand_landmarks_list):
        """Returns True if exactly 2 hands are detected, both fully closed (fists)."""
        if not hand_landmarks_list or len(hand_landmarks_list) != 2:
            return False
        
        for lm in hand_landmarks_list:
            states = self._get_finger_states(lm)
            if states is None: return False
            thumb, index, middle, ring, pinky = states
            if index or middle or ring or pinky:
                return False
        return True

    # ── Tier 3: Extended letter classifier ───────────────────────────────

    def classify_sign(self, hand_landmarks_list, zone="NEUTRAL"):
        """Returns a letter string, or None / '?' if unrecognised.
        zone: suppresses ambiguous letter B when hand is in a word-sign zone.
        """
        if not hand_landmarks_list:
            return None
        lm     = hand_landmarks_list[0]
        states = self._get_finger_states(lm)
        if states is None:
            return None
        thumb, index, middle, ring, pinky = states
        hs = self._hand_size(lm)

        # Suppress all-fingers-open (B) when hand is in a word-sign zone.
        # In those zones, Tier 2 word signs take priority; B as a letter
        # is only meaningful in neutral space.
        word_sign_zones = {"FOREHEAD", "FACE", "CHIN", "CHEST"}
        if all(states) and zone in word_sign_zones:
            return None  # let Tier 2 handle it, don't pollute buffer with B

        if all(states):                                              return "B"
        if not any(states):                                          return "E"
        if not index and not middle and not ring and not pinky:
            return "A" if thumb else "S"

        # O: thumb near index, all moderately curled
        if self._thumb_near(lm, 8, 0.12):
            curls = [self._curl(lm, t, p, m)
                     for t, p, m in [(8,7,5),(12,11,9),(16,15,13),(20,19,17)]]
            if all(c > 0.25 for c in curls):
                return "O"

        # C: all fingers moderately curved, gap between thumb and index
        if not thumb:
            curls = [self._curl(lm, t, p, m)
                     for t, p, m in [(8,7,5),(12,11,9),(16,15,13),(20,19,17)]]
            if all(0.2 < c < 0.75 for c in curls):
                gap = self._dist(lm, 4, 8) / hs
                if 0.25 < gap < 0.65:
                    return "C"

        # F: thumb touches index tip, middle+ring+pinky extended
        if self._thumb_near(lm, 8, 0.10) and middle and ring and pinky:
            return "F"

        # D: index up, thumb near middle
        if index and not middle and not ring and not pinky and self._thumb_near(lm, 12, 0.13):
            return "D"

        # U vs V (index+middle up, close vs spread)
        if index and middle and not ring and not pinky and not thumb:
            gap = abs(lm[8][0] - lm[12][0]) / hs
            return "U" if gap < 0.15 else "V"

        # R: index+middle crossed
        if index and middle and not ring and not pinky:
            if lm[12][0] < lm[8][0] + 0.01:
                return "R"
            return "V"

        # W: 3 fingers
        if index and middle and ring and not pinky:                  return "W"

        # K: index+middle+thumb, thumb between them vertically
        if index and middle and not ring and not pinky and thumb:
            ty = lm[4][1]
            if lm[8][1] < ty < lm[12][1] or lm[12][1] < ty < lm[8][1]:
                return "K"

        # T: thumb tucked between index and middle pips
        if not index and not middle and not ring and not pinky and thumb:
            ty = lm[4][1]
            if min(lm[6][1], lm[10][1]) < ty < max(lm[6][1], lm[10][1]):
                return "T"

        # X: index bent/hooked
        if index and not middle and not ring and not pinky and not thumb:
            if lm[6][1] < lm[8][1] < lm[5][1]:
                return "X"

        if not index and not middle and not ring and pinky:          return "I"
        if thumb and index and not middle and not ring and not pinky: return "L"
        if index and not middle and not ring and not pinky:          return "G"
        if not index and middle and not ring and not pinky:          return "P"
        if thumb and not index and not middle and not ring and pinky: return "Y"
        if index and not middle and not ring and pinky:              return "Y"
        if not thumb and index and middle and ring and pinky:        return "4"

        return "?"

    # ── Main update ───────────────────────────────────────────────────────

    def update(self, hand_landmarks_list, gesture=None, face_zones=None):
        """
        Tier 1 (gesture param) → Tier 2 (word sign).
        Returns the sentence buffer as a string.
        """
        now = time.time()
        self.just_toggled = False

        # --- If Custom Recording Mode is Active ---
        if self._custom_record_word:
            if not hand_landmarks_list:
                self._custom_record_start = None
                return f"RECORDING '{self._custom_record_word}' (No Hand)"
            
            # Update motion buffer to ensure we capture accurate motion
            wrist = hand_landmarks_list[0][0]
            self.motion_buf.add((wrist[0], wrist[1]))
            motion = self.motion_buf.get_motion()
            zone = self._get_zone(wrist[1], face_zones)
            states = self._get_finger_states(hand_landmarks_list[0])

            if self._custom_record_start is None:
                self._custom_record_start = now
            elif (now - self._custom_record_start) >= 1.5:
                # 1.5 seconds have passed with stable hand, save it!
                self._save_custom_gesture(self._custom_record_word, states, zone, motion)
                self._custom_record_word = None
                self._custom_record_start = None
                self._cooldown_until = now + 1.5
                return "SAVED!"
            
            time_left = 1.5 - (now - self._custom_record_start)
            return f"RECORDING '{self._custom_record_word}'... Hold {time_left:.1f}s"

        # --- Check for Start/Stop Toggle Gesture (2 hands open) ---
        if self._is_toggle_gesture(hand_landmarks_list):
            self._comma_start_time = None
            if self._toggle_start_time is None:
                self._toggle_start_time = now
            elif (now - self._toggle_start_time) >= 1.0:
                # Toggle triggered!
                self.is_recording = not self.is_recording
                self.just_toggled = True
                self._toggle_start_time = None
                
                if self.is_recording:
                    self.clear_buffer()  # start fresh when recording begins
                
                # Block sign processing for a moment after toggling
                self._cooldown_until = now + 1.5 
                return " ".join(self.sentence_buffer)
            
            # While holding the toggle gesture, don't process other signs
            return " ".join(self.sentence_buffer)
        else:
            self._toggle_start_time = None

        # --- If paused, ignore all signs ---
        if not self.is_recording:
            return " ".join(self.sentence_buffer)

        # --- Check for Comma Gesture (2 hands closed) ---
        if self._is_comma_gesture(hand_landmarks_list):
            if self._comma_start_time is None:
                self._comma_start_time = now
            elif (now - self._comma_start_time) >= 0.8:
                # Append comma to the buffer if not already ending in one
                if self.sentence_buffer and self.sentence_buffer[-1] != ",":
                    self.sentence_buffer.append(",")
                self._comma_start_time = None
                self._cooldown_until = now + 1.5
                return " ".join(self.sentence_buffer)
            
            # While holding comma gesture, don't process other signs
            return " ".join(self.sentence_buffer)
        else:
            self._comma_start_time = None

        # --- Cooldown: block new signs for SIGN_COOLDOWN_SECONDS after one registers ---
        if now < self._cooldown_until:
            return " ".join(self.sentence_buffer)

        # Update motion buffer
        if hand_landmarks_list:
            wrist = hand_landmarks_list[0][0]
            self.motion_buf.add((wrist[0], wrist[1]))
        motion = self.motion_buf.get_motion()

        # Determine body zone for this frame
        zone = "NEUTRAL"
        if hand_landmarks_list:
            zone = self._get_zone(hand_landmarks_list[0][0][1], face_zones)

        sign = None
        if hand_landmarks_list:
            # Tier 2: Specific position + motion word signs (Highest Priority)
            lm     = hand_landmarks_list[0]
            states = self._get_finger_states(lm)
            sign   = self._classify_word_sign(lm, states, zone, motion)

        # Tier 1: Fallback to basic MediaPipe built-in gesture if Tier 2 didn't match
        if not sign and gesture:
            sign = gesture
            # Tier 3 (letters) disabled per user request

        # Hold-time gating (time-based)
        if sign and sign != "?":
            if sign == self.last_sign:
                # Same sign still held — check if 2 seconds have elapsed
                held = now - (self._sign_start_time or now)
                if held >= config.SIGN_HOLD_SECONDS and not self.word_confirmed:
                    self.sentence_buffer.append(sign)
                    self.word_confirmed  = True
                    self._cooldown_until = now + config.SIGN_COOLDOWN_SECONDS
            else:
                # New sign started
                self.last_sign        = sign
                self._sign_start_time = now
                self.word_confirmed   = False
        else:
            # No sign detected — reset
            self.last_sign        = None
            self._sign_start_time = None
            self.word_confirmed   = False

        return " ".join(self.sentence_buffer)

    # ── Buffer management ─────────────────────────────────────────────────

    def clear_buffer(self):
        self.sentence_buffer  = []
        self.last_sign        = None
        self._sign_start_time = None
        self._cooldown_until  = 0.0
        self._toggle_start_time = None
        self._comma_start_time = None
        self.word_confirmed   = False
        self._prev_buffer_len = 0
        self.motion_buf.reset()

    def get_current_sign(self):
        return self.last_sign

    def get_buffer(self):
        return list(self.sentence_buffer)
