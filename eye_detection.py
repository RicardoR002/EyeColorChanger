import cv2
import numpy as np
import mediapipe as mp

class EyeDetector:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        # Adjust parameters for older MediaPipe version
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
        # Eye landmarks indices in MediaPipe Face Mesh
        # Left eye
        self.LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        # Right eye
        self.RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
    def detect_eyes_mediapipe(self, image):
        """
        Detect eyes using MediaPipe Face Mesh
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            left_eye_mask: Binary mask for left eye
            right_eye_mask: Binary mask for right eye
            success: Boolean indicating if detection was successful
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Create empty masks
        left_eye_mask = np.zeros((h, w), dtype=np.uint8)
        right_eye_mask = np.zeros((h, w), dtype=np.uint8)
        
        try:
            # Process the image
            results = self.face_mesh.process(image_rgb)
            
            if not results.multi_face_landmarks:
                return left_eye_mask, right_eye_mask, False
            
            # Get the first face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract eye landmarks
            left_eye_points = []
            right_eye_points = []
            
            for idx in self.LEFT_EYE_INDICES:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                left_eye_points.append((x, y))
                
            for idx in self.RIGHT_EYE_INDICES:
                landmark = face_landmarks.landmark[idx]
                x, y = int(landmark.x * w), int(landmark.y * h)
                right_eye_points.append((x, y))
            
            # Create masks
            left_eye_points = np.array(left_eye_points, dtype=np.int32)
            right_eye_points = np.array(right_eye_points, dtype=np.int32)
            
            cv2.fillPoly(left_eye_mask, [left_eye_points], 255)
            cv2.fillPoly(right_eye_mask, [right_eye_points], 255)
            
            return left_eye_mask, right_eye_mask, True
        except Exception as e:
            print(f"Error in MediaPipe detection: {e}")
            return left_eye_mask, right_eye_mask, False
    
    def detect_eyes_haarcascade(self, image):
        """
        Detect eyes using Haar Cascade
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            left_eye_mask: Binary mask for left eye
            right_eye_mask: Binary mask for right eye
            success: Boolean indicating if detection was successful
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Create empty masks
        left_eye_mask = np.zeros((h, w), dtype=np.uint8)
        right_eye_mask = np.zeros((h, w), dtype=np.uint8)
        
        try:
            # Load Haar Cascade classifiers
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return left_eye_mask, right_eye_mask, False
            
            # Process the first face
            x, y, w_face, h_face = faces[0]
            face_roi = gray[y:y+h_face, x:x+w_face]
            
            # Detect eyes within the face
            eyes = eye_cascade.detectMultiScale(face_roi)
            
            if len(eyes) < 2:
                return left_eye_mask, right_eye_mask, False
            
            # Sort eyes by x-coordinate
            eyes = sorted(eyes, key=lambda e: e[0])
            
            # Left eye (in image, right eye of the person)
            ex, ey, ew, eh = eyes[0]
            ex, ey = ex + x, ey + y  # Convert to original image coordinates
            cv2.ellipse(right_eye_mask, (ex + ew//2, ey + eh//2), (ew//2, eh//2), 0, 0, 360, 255, -1)
            
            # Right eye (in image, left eye of the person)
            ex, ey, ew, eh = eyes[1]
            ex, ey = ex + x, ey + y  # Convert to original image coordinates
            cv2.ellipse(left_eye_mask, (ex + ew//2, ey + eh//2), (ew//2, eh//2), 0, 0, 360, 255, -1)
            
            return left_eye_mask, right_eye_mask, True
        except Exception as e:
            print(f"Error in Haar Cascade detection: {e}")
            return left_eye_mask, right_eye_mask, False
    
    def detect_eyes_cnn(self, image):
        """
        Placeholder for CNN-based eye detection
        Currently falls back to MediaPipe for detection
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            left_eye_mask: Binary mask for left eye
            right_eye_mask: Binary mask for right eye
            success: Boolean indicating if detection was successful
        """
        # Fall back to MediaPipe
        return self.detect_eyes_mediapipe(image)
    
    def detect_eyes(self, image, method='mediapipe'):
        """
        Detect eyes using the specified method
        
        Args:
            image: Input image (BGR format)
            method: Detection method ('mediapipe', 'haarcascade', or 'cnn')
            
        Returns:
            left_eye_mask: Binary mask for left eye
            right_eye_mask: Binary mask for right eye
            success: Boolean indicating if detection was successful
        """
        if method == 'mediapipe':
            return self.detect_eyes_mediapipe(image)
        elif method == 'haarcascade':
            return self.detect_eyes_haarcascade(image)
        elif method == 'cnn':
            return self.detect_eyes_cnn(image)
        else:
            raise ValueError(f"Unknown method: {method}") 