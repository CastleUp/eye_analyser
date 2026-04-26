import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from mediapipe.python.solutions import face_mesh

class ModelHandler:
    def __init__(self):
        print("Loading DINOv2 model (ViT-S/14)...")
        # Load the smallest DINOv2 model (ViT-S) for speed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(self.device)
        self.model.eval()
        
        # Standard ViT preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224), # DINOv2 expects 224x224
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # We still need FaceMesh for landmarks (internal to EyeProcessor, but just in case)
        # Note: ModelHandler v2 doesn't need FaceAnalysis/InsightFace
        print(f"DINOv2 loaded on {self.device}")

    def get_embedding(self, eye_img):
        if eye_img is None:
            return None
            
        # Convert BGR to RGB
        eye_img_rgb = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)
        
        # Preprocess for DINOv2
        img_tensor = self.transform(eye_img_rgb).unsqueeze(0).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            # Get the [CLS] token embedding (384 dims)
            embedding = self.model(img_tensor).cpu().numpy().flatten()
            
        # Extract color features (mean BGR of the eye center)
        # Reuse the logic from v1 for consistency
        h, w = eye_img.shape[:2]
        roi = eye_img[h//2-20:h//2+20, w//2-20:w//2+20]
        avg_color = np.mean(roi, axis=(0, 1)) / 255.0
        color_features = np.tile(avg_color, 20) # 60 dimensions
        
        return np.concatenate([embedding, color_features])

    def get_combined_embedding(self, left_eye, right_eye):
        emb_l = self.get_embedding(left_eye)
        emb_r = self.get_embedding(right_eye)
        
        if emb_l is not None and emb_r is not None:
            # Concatenate left and right eye embeddings (Total: (384 + 60) * 2 = 888)
            return np.concatenate([emb_l, emb_r])
        return None
