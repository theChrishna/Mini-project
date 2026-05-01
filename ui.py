import cv2
import numpy as np
import time
import config
from PIL import Image, ImageDraw, ImageFont

# Color palette (BGR)
COLORS = {
    "bg":          (15, 15, 15),
    "accent":      (220, 220, 220),
    "green":       (255, 255, 255),
    "red":         (60, 60, 220),
    "yellow":      (200, 200, 200),
    "white":       (250, 250, 250),
    "dark":        (30, 30, 30),
    "panel_bg":    (20, 20, 20),
}

def bgr2rgb(c):
    """Convert OpenCV BGR to PIL RGB"""
    return (c[2], c[1], c[0])

# Global Fonts setup (Segoe UI is identical to Inter and native to Windows)
try:
    font_xl = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 32)
    font_lg = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 22)
    font_md = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
    font_sm = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 13)
except IOError:
    print("[UI] Segoe UI font not found, using default.")
    font_xl = font_lg = font_md = font_sm = ImageFont.load_default()


def draw_ui_overlay(
    frame,
    sign_buffer: str,
    current_sign: str,
    emotion: str,
    translated_text: str,
    toggles: dict,
    fps: float,
    hold_progress: float = 0.0,
    gesture_conf: float  = 0.0,
    is_recording: bool   = False,
):
    h, w = frame.shape[:2]

    # --- Draw OpenCV Shapes First (Rectangles, Circles, Lines) ---
    
    # Header strip
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), COLORS["bg"], -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Recording Status Dot
    dot_x, dot_y = w - 30, 30
    if is_recording:
        if time.time() % 1.0 < 0.5:
            cv2.circle(frame, (dot_x, dot_y), 8, COLORS["red"], -1)
    else:
        cv2.circle(frame, (dot_x, dot_y), 8, COLORS["white"], -1)

    # Hold-progress bar
    if current_sign:
        bar_max_w = 220
        bar_filled = int(bar_max_w * hold_progress)
        cv2.rectangle(frame, (16, 104), (16 + bar_max_w, 112), COLORS["dark"],   -1)
        cv2.rectangle(frame, (16, 104), (16 + bar_filled, 112), COLORS["accent"], -1)

    # Buffer sentence bar at bottom
    bar_h = 80
    bar_y = h - bar_h
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, bar_y), (w, h), COLORS["panel_bg"], -1)
    cv2.addWeighted(overlay2, 0.82, frame, 0.18, 0, frame)
    cv2.line(frame, (0, bar_y), (w, bar_y), COLORS["accent"], 2)

    # Toggle panel (right side)
    panel_w = 180
    panel_x = w - panel_w - 10
    panel_y = 70
    panel_h = 160
    overlay3 = frame.copy()
    cv2.rectangle(overlay3, (panel_x - 8, panel_y - 8),
                  (panel_x + panel_w, panel_y + panel_h), COLORS["dark"], -1)
    cv2.addWeighted(overlay3, 0.70, frame, 0.30, 0, frame)

    # --- Convert to PIL for beautiful Text Rendering ---
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(pil_img)

    def put_text(text, xy, font, color_key="white"):
        draw.text(xy, text, font=font, fill=bgr2rgb(COLORS.get(color_key, COLORS["white"])))

    # Title
    put_text("Signfy", (16, 12), font_xl, "white")

    # Current detected sign badge
    if current_sign:
        conf_tag = f"  ({gesture_conf*100:.0f}%)" if gesture_conf > 0 else ""
        badge_label = f"Sign: {current_sign}{conf_tag}"
        put_text(badge_label, (16, 75), font_lg, "white")

    # Emotion label
    put_text(f"Emotion: {emotion}", (16, 120), font_md, "accent")

    # FPS
    put_text(f"{fps:.1f} FPS", (w - 75, bar_y - 25), font_sm, "accent")

    # Buffer label & text
    put_text("Buffer:", (12, bar_y + 12), font_sm, "accent")
    put_text(sign_buffer or "--", (80, bar_y + 8), font_md, "white")

    # Translated output label & text
    put_text("Output:", (12, bar_y + 45), font_sm, "green")
    put_text(translated_text or "--", (80, bar_y + 40), font_lg, "white")

    # Settings Panel Header
    put_text("Settings [Keys]", (panel_x, panel_y), font_sm, "accent")

    toggle_items = [
        ("H", "Hand Track", toggles.get("hand")),
        ("E", "Emotion", toggles.get("emotion")),
        ("L", "LLM", toggles.get("llm")),
        ("V", "Voice", toggles.get("voice")),
    ]

    for i, (key, label, state) in enumerate(toggle_items):
        color = "green" if state else "red"
        status = "ON" if state else "OFF"
        y_pos = panel_y + 28 + i * 28
        put_text(f"[{key}] {label}: {status}", (panel_x, y_pos), font_sm, color)

    # Keyboard hint
    put_text("[C] Clear  [Q] Quit  [S] Speak", (10, h - bar_h - 25), font_sm, "white")

    # --- Convert back to OpenCV BGR ---
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
