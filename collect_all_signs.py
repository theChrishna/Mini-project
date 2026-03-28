import cv2
import csv
import os
import time
from hand_tracking import HandDetector

def main():
    cap = cv2.VideoCapture(0)
    detector = HandDetector(maxHands=1)
    file_name = "hand_data.csv"
    
    # Create the file with a header if it doesn't exist
    if not os.path.exists(file_name):
        with open(file_name, 'w', newline='') as f:
            header = ['label'] + [f'pt{i}_x' for i in range(21)] + [f'pt{i}_y' for i in range(21)]
            csv.writer(f).writerow(header)

    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    frames_per_sign = 150 # Taking 150 snapshot frames per sign to build a massive robust dataset
    
    print("\n" + "="*50)
    print("Welcome to the Ultimate A-Z Data Collector!")
    print("We will rapidly collect high-quality data for all 26 letters.")
    print("="*50)
    input("Press ENTER to start the collection process...")
    
    for label in alphabet:
        print(f"\n--- Prepare to show the sign for '{label}' ---")
        for i in range(3, 0, -1):
            print(f"Starting in {i}...")
            time.sleep(1)
            
        print(f"🔴 RECORDING {frames_per_sign} frames for '{label}'! Perform micro-movements to capture variation.")
        
        count = 0
        while count < frames_per_sign:
            success, img = cap.read()
            if not success: continue
                
            img = cv2.flip(img, 1)    
            img = detector.findHands(img, 0, draw=True)
            lmList = detector.findPosition(img, draw=False)
            
            if len(lmList) != 0:
                h, w, c = img.shape
                x_coords = [lm[1]/w for lm in lmList]
                y_coords = [lm[2]/h for lm in lmList]
                
                row = [label] + x_coords + y_coords
                with open(file_name, 'a', newline='') as f:
                    csv.writer(f).writerow(row)
                count += 1
                
                # Visually show user progress on-screen
                cv2.putText(img, f"Recording '{label}': {int((count/frames_per_sign)*100)}%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.putText(img, "Hand not detected! Bring hand into frame.", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            cv2.imshow("Mass Data Collection", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Process interrupted by user.")
                return

    print("\n✅ MASS DATA COLLECTION COMPLETE!")
    print("You now have data for all 26 letters of the alphabet.")
    print("Run `train_model.py` next to build the new deep-learning model!")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
