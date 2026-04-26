import cv2
import numpy as np
import time
import csv
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Add subfolders to path to allow imports
sys.path.append(os.path.join(os.getcwd(), 'v1_arcface'))
import v1_arcface.eye_processor as ep
import v1_arcface.model_handler as mh1
import v1_arcface.db_handler as db1

sys.path.append(os.path.join(os.getcwd(), 'v2_dinov2'))
import v2_dinov2.model_handler as mh2
import v2_dinov2.db_handler as db2

def run_comparison():
    print("Initializing Battle of Algorithms...")
    
    # Initialize V1
    processor = ep.EyeProcessor()
    model1 = mh1.ModelHandler()
    database1 = db1.DBHandler()
    
    # Initialize V2
    model2 = mh2.ModelHandler()
    database2 = db2.DBHandler()
    
    cap = cv2.VideoCapture(0)
    
    # Logging
    log_file = open('comparison_log.csv', 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['Timestamp', 'V1_Distance', 'V2_Distance', 'V1_Match', 'V2_Match'])
    
    # Real-time data for plotting
    v1_dists = []
    v2_dists = []
    timestamps = []
    start_time = time.time()
    
    print("Comparison started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()
        left_eye, right_eye, landmarks = processor.get_eye_crops(frame)
        
        v1_dist = 1.0
        v2_dist = 1.0
        v1_name = "Unknown"
        v2_name = "Unknown"
        
        if left_eye is not None and right_eye is not None:
            # Model 1 (ArcFace)
            emb1 = model1.get_combined_embedding(left_eye, right_eye)
            if emb1 is not None:
                name1, d1 = database1.query_user(emb1, threshold=1.0) # No threshold for logging
                v1_dist = d1
                v1_name = name1 if name1 else "Unknown"
            
            # Model 2 (DINOv2)
            emb2 = model2.get_combined_embedding(left_eye, right_eye)
            if emb2 is not None:
                name2, d2 = database2.query_user(emb2, threshold=1.0)
                v2_dist = d2
                v2_name = name2 if name2 else "Unknown"
            
            # Log results
            curr_ts = time.time() - start_time
            log_writer.writerow([curr_ts, v1_dist, v2_dist, v1_name, v2_name])
            
            v1_dists.append(v1_dist)
            v2_dists.append(v2_dist)
            timestamps.append(curr_ts)
            
            # Keep only last 50 points for display
            if len(v1_dists) > 50:
                v1_dists.pop(0)
                v2_dists.pop(0)
                timestamps.pop(0)
                
            processor.draw_landmarks(display_frame, landmarks)
            
        # UI Overlay
        cv2.putText(display_frame, f"V1 (ArcFace): {v1_name} ({v1_dist:.3f})", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_frame, f"V2 (DINOv2):  {v2_name} ({v2_dist:.3f})", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("Battle of Algorithms", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    log_file.close()
    print(f"Comparison log saved to comparison_log.csv")

if __name__ == "__main__":
    run_comparison()
