import cv2
import numpy as np
import os
import sys

# Windows CUDA DLL Fix for onnxruntime-gpu
if sys.platform == 'win32':
    import site
    # Check both system and user site-packages
    for p in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia_base = os.path.join(p, 'nvidia')
        if os.path.exists(nvidia_base):
            for sub in ['cublas', 'cudnn', 'cuda_runtime', 'cuda_nvrtc']:
                bin_path = os.path.join(nvidia_base, sub, 'bin')
                if os.path.exists(bin_path):
                    try:
                        # os.add_dll_directory is for Python 3.8+
                        os.add_dll_directory(bin_path)
                        # Some versions of ORT also need it in PATH
                        os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']
                    except Exception:
                        pass

import insightface
from insightface.app import FaceAnalysis

class ModelHandler:
    def __init__(self):
        # Initialize FaceAnalysis with the buffalo_l model
        # Using DmlExecutionProvider (DirectML) as it is most reliable for Windows GPUs
        self.app = FaceAnalysis(name='buffalo_l', providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # We might want to use the recognition model directly for crops
        self.rec_model = self.app.models.get('recognition')

    def get_embedding(self, eye_img):
        if eye_img is None:
            return None
            
        if eye_img.shape[:2] != (112, 112):
            eye_img = cv2.resize(eye_img, (112, 112))
            
        # Get neural embedding
        embedding = self.rec_model.get_feat(eye_img).flatten()
        
        # Extract color features (mean BGR of the eye center)
        # Focus on the 40x40 area in the center of the 112x112 image
        h, w = eye_img.shape[:2]
        roi = eye_img[h//2-20:h//2+20, w//2-20:w//2+20]
        avg_color = np.mean(roi, axis=(0, 1)) / 255.0 # Normalize 0-1
        
        # We give color features a strong signal by repeating them
        color_features = np.tile(avg_color, 20) # 60 dimensions
        
        # Concatenate neural features (512) with color features (60)
        return np.concatenate([embedding, color_features])

    def get_combined_embedding(self, left_eye, right_eye):
        emb_l = self.get_embedding(left_eye)
        emb_r = self.get_embedding(right_eye)
        
        if emb_l is not None and emb_r is not None:
            # Concatenate left and right eye embeddings (Total: 572 * 2 = 1144)
            return np.concatenate([emb_l, emb_r])
        return None
