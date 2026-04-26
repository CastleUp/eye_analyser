import cv2
from mediapipe.python.solutions import face_mesh as mp_face_mesh
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
import numpy as np

class EyeProcessor:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Landmark indices for eyes
        # Left eye: 33, 133, 160, 158, 153, 144, 163, 7
        # Right eye: 362, 263, 387, 385, 373, 380, 390, 249
        self.LEFT_EYE_INDEXES = [33, 133, 160, 158, 153, 144, 163, 7]
        self.RIGHT_EYE_INDEXES = [362, 263, 387, 385, 373, 380, 390, 249]

    def get_eye_crops(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, None, None
            
        landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        
        left_eye_points = []
        for idx in self.LEFT_EYE_INDEXES:
            pt = landmarks.landmark[idx]
            left_eye_points.append([int(pt.x * w), int(pt.y * h)])
            
        right_eye_points = []
        for idx in self.RIGHT_EYE_INDEXES:
            pt = landmarks.landmark[idx]
            right_eye_points.append([int(pt.x * w), int(pt.y * h)])
            
        left_eye_crop = self._crop_eye(frame, left_eye_points)
        right_eye_crop = self._crop_eye(frame, right_eye_points)
        
        return left_eye_crop, right_eye_crop, landmarks

    def _crop_eye(self, frame, points, padding=0.7):
        points = np.array(points)
        x, y, w, h = cv2.boundingRect(points)
        
        # Add padding to capture more context (eyebrows, etc.)
        pw = int(w * padding)
        ph = int(h * padding)
        
        x1 = max(0, x - pw)
        y1 = max(0, y - ph)
        x2 = min(frame.shape[1], x + w + pw)
        y2 = min(frame.shape[0], y + h + ph)
        
        eye_img = frame[y1:y2, x1:x2]
        if eye_img.size == 0:
            return None
            
        # Standardize size
        eye_img = cv2.resize(eye_img, (112, 112))
        
        # Image Enhancement: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # This helps in highlighting the iris and texture details around the eye
        lab = cv2.cvtColor(eye_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        eye_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        return eye_img

    def draw_landmarks(self, frame, landmarks):
        # Optional: draw landmarks for debugging
        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
