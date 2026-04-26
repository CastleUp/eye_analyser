import cv2
import sys
import os
from eye_processor import EyeProcessor
from model_handler import ModelHandler
from db_handler import DBHandler

def enroll_from_image(image_path, name):
    if not os.path.exists(image_path):
        print(f"Error: File {image_path} not found.")
        return

    processor = EyeProcessor()
    model = ModelHandler()
    db = DBHandler()

    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not read image.")
        return

    left_eye, right_eye, landmarks = processor.get_eye_crops(frame)
    
    if left_eye is not None and right_eye is not None:
        combined_emb = model.get_combined_embedding(left_eye, right_eye)
        if combined_emb is not None:
            # We don't clear the database here, just append
            db.collection.add(
                embeddings=[combined_emb.tolist()],
                ids=[name],
                metadatas=[{"name": name}]
            )
            print(f"Successfully enrolled {name} from image!")
        else:
            print("Error: Could not generate embedding.")
    else:
        print("Error: Could not detect eyes in the image.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python enroll_image.py <image_path> <name>")
    else:
        enroll_from_image(sys.argv[1], sys.argv[2])
