import cv2
import numpy as np
from eye_processor import EyeProcessor
from model_handler import ModelHandler
from db_handler import DBHandler
import time

def enroll_user():
    db = DBHandler()
    
    choice = input("Do you want to clear the database before enrollment? (y/n): ").lower()
    if choice == 'y':
        print("Clearing previous records...")
        db.clear_database()
    
    name = input("Enter the name of the person to enroll: ")
    if not name:
        print("Name cannot be empty.")
        return

    processor = EyeProcessor()
    model = ModelHandler()

    cap = cv2.VideoCapture(0)
    
    embeddings = []
    required_frames = 20 # Increased for better accuracy
    capturing = False
    
    print(f"\nInstructions for {name}:")
    print("1. Look straight into the camera.")
    print("2. Ensure eyes are wide open.")
    print("3. Press 'SPACE' to start capturing when you are ready.")
    print("4. Press 'q' to cancel.")
    
    while len(embeddings) < required_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        display_frame = frame.copy()
        left_eye, right_eye, landmarks = processor.get_eye_crops(frame)
        
        # Check if face is close enough (simple heuristic: distance between eyes)
        face_is_ready = False
        if landmarks:
            # Distance between inner eye corners
            l_inner = landmarks.landmark[133]
            r_inner = landmarks.landmark[362]
            dist = np.sqrt((l_inner.x - r_inner.x)**2 + (l_inner.y - r_inner.y)**2)
            
            if dist > 0.15: # Face is close enough
                face_is_ready = True
                msg = "READY TO CAPTURE"
                color = (0, 255, 0)
            else:
                msg = "TOO FAR! MOVE CLOSER"
                color = (0, 0, 255)
        else:
            msg = "FACE NOT DETECTED"
            color = (0, 0, 255)

        if not capturing:
            cv2.putText(display_frame, f"PRESS SPACE TO START | {msg}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(display_frame, f"CAPTURING: {len(embeddings)}/{required_frames}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if left_eye is not None and right_eye is not None and face_is_ready:
                combined_emb = model.get_combined_embedding(left_eye, right_eye)
                if combined_emb is not None:
                    embeddings.append(combined_emb)
                    
        # Show eye crops preview
        if left_eye is not None and right_eye is not None:
            display_frame[50:50+112, 20:20+112] = left_eye
            display_frame[50:50+112, 140:140+112] = right_eye

        cv2.imshow("Interactive Enrollment", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            if face_is_ready:
                capturing = True
            else:
                print("Please move closer before starting.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) == required_frames:
        avg_embedding = np.mean(embeddings, axis=0)
        db.add_user(name, avg_embedding)
        print(f"Successfully enrolled {name}!")
    else:
        print("Enrollment failed or cancelled.")

if __name__ == "__main__":
    enroll_user()
