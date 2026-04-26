import cv2
import numpy as np
from eye_processor import EyeProcessor
from model_handler import ModelHandler
from db_handler import DBHandler
import time

def run_recognition():
    processor = EyeProcessor()
    model = ModelHandler()
    db = DBHandler()

    cap = cv2.VideoCapture(0)
    
    print("Starting real-time eye recognition...")
    print("Press 'q' to quit.")

    # Performance monitoring
    fps_time = time.time()
    
    # Threshold - lowered for better accuracy (stricter)
    # Threshold - even stricter now
    THRESHOLD = 0.28
    
    # History for temporal voting (smoothing results)
    history = []
    HISTORY_SIZE = 10
    confirmed_name = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        display_frame = frame.copy()
        left_eye, right_eye, landmarks = processor.get_eye_crops(frame)
        
        current_name = None
        status_text = "SCANNING..."
        status_color = (255, 255, 0)

        if left_eye is not None and right_eye is not None:
            combined_emb = model.get_combined_embedding(left_eye, right_eye)
            
            if combined_emb is not None:
                name, distance = db.query_user(combined_emb, threshold=THRESHOLD)
                current_name = name if name else "Unknown"
                
                # Debug logging
                if name:
                    print(f"Match found: {name} | Distance: {distance:.4f}")
                else:
                    print(f"Unknown user | Distance: {distance:.4f}")
                
                # Update history
                history.append(current_name)
                if len(history) > HISTORY_SIZE:
                    history.pop(0)
                
                # Voting logic
                if len(history) == HISTORY_SIZE:
                    # Count occurrences of each name
                    counts = {}
                    for n in history:
                        counts[n] = counts.get(n, 0) + 1
                    
                    # Find most frequent name
                    best_name = max(counts, key=counts.get)
                    # Requirement: name must appear in > 60% of frames
                    if counts[best_name] > (HISTORY_SIZE * 0.6) and best_name != "Unknown":
                        confirmed_name = best_name
                        status_text = f"SUCCESS: {confirmed_name}"
                        status_color = (0, 255, 0)
                    else:
                        confirmed_name = None
                        status_text = "ACCESS FORBIDDEN"
                        status_color = (0, 0, 255)
                
                processor.draw_landmarks(display_frame, landmarks)
        else:
            history = [] # Reset history if face is lost
            status_text = "EYES NOT DETECTED"
            status_color = (0, 255, 255)

        # UI Overlay
        cv2.putText(display_frame, status_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # Show eye crops in corner
        if left_eye is not None and right_eye is not None:
            display_frame[50:50+112, 20:20+112] = left_eye
            display_frame[50:50+112, 140:140+112] = right_eye

        cv2.imshow("Eye Recognizer", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition()
