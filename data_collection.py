import cv2
import csv
import os
from hand_tracking import HandDetector

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    
    # --- DATA COLLECTION SETTINGS ---
    label = "C"  # Change this to "B", "C", etc. before running
    file_name = "hand_data.csv"
    # Create the file with a header if it doesn't exist
    if not os.path.exists(file_name):
        with open(file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['label'] + [f'pt{i}_x' for i in range(21)] + [f'pt{i}_y' for i in range(21)]
            writer.writerow(header)
    # --------------------------------

    print(f"Recording for letter: {label}")
    print("Press 's' to SAVE a frame | Press 'q' to QUIT")

    frame_timestamp_ms = 0

    while True:
        success, img = cap.read()
        if not success: continue
            
        img = cv2.flip(img, 1)    
        fps_prop = cap.get(cv2.CAP_PROP_FPS)
        frame_timestamp_ms += int(1000 / fps_prop) if fps_prop > 0 else 30
        
        img = detector.findHands(img, frame_timestamp_ms, draw=True)
        lmList = detector.findPosition(img, draw=False)
        
        if len(lmList) != 0:
            # Check for 's' key to save data
            key = cv2.waitKey(1)
            if key & 0xFF == ord('s'):
                # Prepare our data row: [Label, x0, x1...x20, y0, y1...y20]
                # We normalize coordinates (dividing by width/height) so the AI 
                # works even if your hand is closer or further from the camera.
                h, w, c = img.shape
                x_coords = [lm[1]/w for lm in lmList]
                y_coords = [lm[2]/h for lm in lmList]
                
                row = [label] + x_coords + y_coords
                
                with open(file_name, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                
                print(f"Saved data for {label}!")
                # Visual feedback on screen
                cv2.putText(img, "SAVED!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Data Collection", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()