import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import os
import time
from collections import deque
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from alert_system import AlertSystem

class DrowsinessDetector:
    """Advanced Driver Monitoring System (DMS) identifying drowsiness, yawns, and distraction."""

    def __init__(self, model_path='models/alertdrive_best_model.h5'):
        """Initialize the expanded DMS detector."""
        print(f"Loading model from {model_path}...")
        try:
            self.model = load_model(model_path)
            print("CNN Model loaded successfully.")
            self.use_model = True
        except Exception as e:
            print(f"Failed to load CNN model: {e}. Falling back to geometric detection...")
            self.use_model = False
            
        self.consecutive_drowsy_frames = 0
        self.drowsy_threshold = 12

        self.consecutive_distracted_frames = 0
        self.distraction_threshold = 15 # Approx 1-2 seconds of looking away

        # Initialize the Alert System
        self.alert_system = AlertSystem()

        # -----------------------------
        # 1. MediaPipe FaceLandmarker Initialization (Tasks API)
        # -----------------------------
        # We use the new Tasks API because mp.solutions is deprecated on Python 3.12+
        base_options = mp_python.BaseOptions(model_asset_path='models/face_landmarker.task')
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        
        # -----------------------------
        # 2. DMS Feature Buffers & Thresholds
        # -----------------------------
        self.ear_buffer = deque(maxlen=10) # Anti-flicker for eyes
        self.mar_buffer = deque(maxlen=10) # Anti-flicker for mouth
        
        self.MAR_THRESHOLD = 0.5  # Threshold for Yawn detection
        self.EAR_THRESHOLD = 0.22 # Default baseline, overridden by auto-calibration
        
        # MediaPipe Indices
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        
        # -----------------------------
        # 3. Auto-Calibration Phase Setup
        # -----------------------------
        self.is_calibrating = True
        self.calibration_frames = 0
        self.calibration_max_frames = 50 # ~3 seconds of startup calibration
        self.calibration_ear_sum = 0.0

    def _apply_clahe(self, frame):
        """Applies CLAHE for dynamic car lighting correction."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def _calculate_ear(self, landmarks, eye_indices, iw, ih):
        """Calculates Eye Aspect Ratio (EAR)."""
        points = [np.array([landmarks[i].x * iw, landmarks[i].y * ih]) for i in eye_indices]
        v_dist1 = np.linalg.norm(points[1] - points[5])
        v_dist2 = np.linalg.norm(points[2] - points[4])
        h_dist = np.linalg.norm(points[0] - points[3])
        return (v_dist1 + v_dist2) / (2.0 * h_dist)

    def _calculate_mar(self, landmarks, iw, ih):
        """Calculates Mouth Aspect Ratio (MAR) for Yawn Detection."""
        # Inner lip coordinates
        top = np.array([landmarks[13].x * iw, landmarks[13].y * ih])
        bottom = np.array([landmarks[14].x * iw, landmarks[14].y * ih])
        left = np.array([landmarks[78].x * iw, landmarks[78].y * ih])
        right = np.array([landmarks[308].x * iw, landmarks[308].y * ih])
        
        v_dist = np.linalg.norm(top - bottom)
        h_dist = np.linalg.norm(left - right)
        return v_dist / (h_dist + 1e-6)

    def _check_distraction(self, landmarks, iw, ih):
        """Calculates Head Pose (Pitch & Yaw) via proportional geometric distances."""
        nose = np.array([landmarks[1].x * iw, landmarks[1].y * ih])
        
        # YAW (Looking Left/Right)
        left_edge = np.array([landmarks[234].x * iw, landmarks[234].y * ih])
        right_edge = np.array([landmarks[454].x * iw, landmarks[454].y * ih])
        left_dist = np.linalg.norm(nose - left_edge)
        right_dist = np.linalg.norm(nose - right_edge)
        yaw_ratio = left_dist / (right_dist + 1e-6)
        
        # PITCH (Looking Up/Down)
        top_edge = np.array([landmarks[10].x * iw, landmarks[10].y * ih])
        bottom_edge = np.array([landmarks[152].x * iw, landmarks[152].y * ih])
        top_dist = np.linalg.norm(nose - top_edge)
        bot_dist = np.linalg.norm(nose - bottom_edge)
        pitch_ratio = top_dist / (bot_dist + 1e-6)
        
        is_distracted = False
        # Head tilted far left or right (Adjusted for better real-world tolerance)
        if yaw_ratio < 0.55 or yaw_ratio > 1.8:
            is_distracted = True
        # Head pitched heavily down (e.g. looking at phone) or up
        if pitch_ratio > 1.4 or pitch_ratio < 0.7:
            is_distracted = True
            
        return is_distracted

    def run(self):
        """Run real-time exhaustive DMS detection using webcam."""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Starting Driver Monitoring System... Press 'Q' to quit.")
        
        class MockLandmarks:
            """Bridge class to map Tasks API arrays to my old attributes."""
            def __init__(self, lmks):
                self.landmark = lmks

        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            ih, iw, _ = frame.shape
            enhanced_frame = self._apply_clahe(frame)
            rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
            
            # New MediaPipe Tasks inference
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.face_landmarker.detect(mp_image)
            
            probability = 0.5
            smoothed_ear = 0.0
            smoothed_mar = 0.0
            status = "NATURAL"
            color = (0, 255, 0)
            is_alert_triggered = False
            
            if results.face_landmarks:
                # Wrap the new landmarks in the compatible MockLandmarks object
                face_landmarks = MockLandmarks(results.face_landmarks[0])
                
                # --- Geometric Metric Gathering ---
                left_ear = self._calculate_ear(face_landmarks.landmark, self.LEFT_EYE, iw, ih)
                right_ear = self._calculate_ear(face_landmarks.landmark, self.RIGHT_EYE, iw, ih)
                avg_ear = (left_ear + right_ear) / 2.0
                mar = self._calculate_mar(face_landmarks.landmark, iw, ih)
                
                self.ear_buffer.append(avg_ear)
                self.mar_buffer.append(mar)
                smoothed_ear = sum(self.ear_buffer) / len(self.ear_buffer)
                smoothed_mar = sum(self.mar_buffer) / len(self.mar_buffer)
                
                # --- Phase 1: Auto-Calibration ---
                if self.is_calibrating:
                    self.calibration_ear_sum += avg_ear
                    self.calibration_frames += 1
                    
                    # Display Calibration UI
                    cv2.putText(frame, f"CALIBRATING EYES...", (50, int(ih/2) - 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                    cv2.putText(frame, f"Please look forward: {int((self.calibration_frames/self.calibration_max_frames)*100)}%", 
                                (50, int(ih/2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    
                    if self.calibration_frames >= self.calibration_max_frames:
                        baseline_ear = self.calibration_ear_sum / self.calibration_max_frames
                        # Set threshold to 75% of driver's normal open-eye baseline
                        self.EAR_THRESHOLD = baseline_ear * 0.75 
                        self.is_calibrating = False
                        print(f"Calibration Complete! Base EAR: {baseline_ear:.3f}. Threshold locked at: {self.EAR_THRESHOLD:.3f}")
                    
                    cv2.imshow("AlertDrive Advanced DMS", frame)
                    cv2.waitKey(1)
                    continue # Skip alert logic while calibrating
                
                # --- Phase 2: CNN Backup Predictor ---
                if self.use_model:
                    x_min = min([lm.x for lm in face_landmarks.landmark]) * iw
                    x_max = max([lm.x for lm in face_landmarks.landmark]) * iw
                    y_min = min([lm.y for lm in face_landmarks.landmark]) * ih
                    y_max = max([lm.y for lm in face_landmarks.landmark]) * ih
                    
                    x, y = int(x_min - 20), int(y_min - 20)
                    w, h_box = int(x_max - x_min + 40), int(y_max - y_min + 40)
                    x, y = max(0, x), max(0, y)
                    w, h_box = min(iw - x, w), min(ih - y, h_box)
                    
                    face_crop = rgb_frame[y:y+h_box, x:x+w]
                    if face_crop.size > 0:
                        resized_frame = cv2.resize(face_crop, (224, 224))
                        input_frame = np.expand_dims(resized_frame, axis=0)
                        formatted_input = preprocess_input(input_frame.astype(np.float32))
                        prediction = self.model.predict(formatted_input, verbose=0)
                        probability = prediction[0][0]

                # --- Phase 3: Core Decision Logic ---
                is_yawning = smoothed_mar > self.MAR_THRESHOLD
                is_distracted = self._check_distraction(face_landmarks.landmark, iw, ih)
                
                # Prevent false drowsy triggers when head is turned (EAR gets skewed on side profiles)
                if is_distracted:
                    is_drowsy = False
                else:
                    is_drowsy_ear = (smoothed_ear < self.EAR_THRESHOLD)
                    if self.use_model:
                        # A common issue is the CNN model falsely predicting "drowsy" even when eyes are wide open.
                        # Geometric EAR is absolute. If EAR is clearly above the threshold (eyes visibly open),
                        # we must ignore any false positive CNN predictions.
                        if smoothed_ear > (self.EAR_THRESHOLD * 1.15):
                            is_drowsy = False
                        else:
                            is_drowsy = is_drowsy_ear or (probability < 0.3)
                    else:
                        is_drowsy = is_drowsy_ear
                
                # Smooth Counter Updates (gradual decay prevents micro-glitches from resetting completely)
                if is_distracted:
                    self.consecutive_distracted_frames += 1
                else:
                    self.consecutive_distracted_frames = max(0, self.consecutive_distracted_frames - 2)
                    
                if is_drowsy:
                    self.consecutive_drowsy_frames += 1
                else:
                    self.consecutive_drowsy_frames = max(0, self.consecutive_drowsy_frames - 2)
                
                # State Machine Priorities: Evaluate statuses based strictly on thresholds
                if self.consecutive_drowsy_frames >= self.drowsy_threshold:
                    status = "DROWSY"
                    color = (0, 0, 255) # RED
                    is_alert_triggered = True
                elif self.consecutive_distracted_frames >= self.distraction_threshold:
                    status = "DISTRACTED: EYES ON ROAD!"
                    color = (0, 165, 255) # ORANGE
                    is_alert_triggered = True
                elif is_yawning:
                    status = "EARLY WARNING: YAWN DETECTED"
                    color = (0, 255, 255) # YELLOW
                else:
                    status = "NATURAL"
                    color = (0, 255, 0) # GREEN
            else:
                self.consecutive_drowsy_frames = max(0, self.consecutive_drowsy_frames - 1)
                self.consecutive_distracted_frames = max(0, self.consecutive_distracted_frames - 1)
            
            # --- Phase 4: Warning Integrations ---
            
            # Pass to audio subsystem
            self.alert_system.process_frame(status, is_alert_triggered)
            display_msg = self.alert_system.get_display_message(status)
            
            # Draw primary bounding rectangle
            cv2.rectangle(frame, (0, 0), (iw, ih), color, 6)
            
            # Draw active telemetry metrics
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"EAR: {smoothed_ear:.3f} / {self.EAR_THRESHOLD:.3f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"MAR: {smoothed_mar:.3f} / {self.MAR_THRESHOLD:.3f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            if is_alert_triggered:
                alert_text = "!!! WAKE UP !!!" if "DROWSY" in status else "!!! EYES ON ROAD !!!"
                cv2.putText(frame, alert_text, (50, int(ih / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            cv2.imshow("AlertDrive Advanced DMS", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.alert_system.stop_alert()
        print(self.alert_system.generate_report())

if __name__ == "__main__":
    detector = DrowsinessDetector()
    detector.run()
