"""
Run this script to find which camera index your DroidCam uses.
It will try indices 0–5 and show what's available.
"""
import cv2

print("Scanning camera indices 0–5...\n")
found = []

for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # CAP_DSHOW = Windows DirectShow
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            print(f"  [FOUND] Index {i} — {w}x{h} — showing preview for 2 seconds")
            cv2.imshow(f"Camera {i} — press any key to close", frame)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()
            found.append(i)
        else:
            print(f"  [EXISTS but no frame] Index {i}")
        cap.release()
    else:
        print(f"  [NOT FOUND] Index {i}")

print(f"\nAvailable cameras: {found}")
print(f"Set CAMERA_INDEX in config.py to whichever index showed your phone screen.")
