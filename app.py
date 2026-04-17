import gradio as gr
import cv2
import numpy as np
from detector import DrowsinessDetector
import os

# Initialize the detector
# Note: We initialize it here, but in a multi-user environment, 
# you might want to move this inside the function or use gr.State
detector = DrowsinessDetector(model_path='models/alertdrive_best_model.h5')

def predict(frame):
    if frame is None:
        return None, "No frame detected"

    # Convert RGB (Gradio) to BGR (OpenCV)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ih, iw, _ = frame_bgr.shape
    
    # --- Integration logic from detector.py ---
    # We replicate the core logic of detector.run() but for a single frame
    enhanced_frame = detector._apply_clahe(frame_bgr)
    rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
    
    # MediaPipe Tasks inference
    import mediapipe as mp
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.face_landmarker.detect(mp_image)
    
    status = "NATURAL"
    color = (0, 255, 0)
    is_alert_triggered = False
    
    # Bridge class to map landmarks (same as in detector.py)
    class MockLandmarks:
        def __init__(self, lmks):
            self.landmark = lmks

    if results.face_landmarks:
        face_landmarks = MockLandmarks(results.face_landmarks[0])
        
        # Calibration check
        if detector.is_calibrating:
            left_ear = detector._calculate_ear(face_landmarks.landmark, detector.LEFT_EYE, iw, ih)
            right_ear = detector._calculate_ear(face_landmarks.landmark, detector.RIGHT_EYE, iw, ih)
            avg_ear = (left_ear + right_ear) / 2.0
            
            detector.calibration_ear_sum += avg_ear
            detector.calibration_frames += 1
            
            progress = int((detector.calibration_frames / detector.calibration_max_frames) * 100)
            cv2.putText(frame_bgr, f"CALIBRATING: {progress}%", (50, ih // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            
            if detector.calibration_frames >= detector.calibration_max_frames:
                baseline_ear = detector.calibration_ear_sum / detector.calibration_max_frames
                detector.EAR_THRESHOLD = baseline_ear * 0.75 
                detector.is_calibrating = False
            
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), "Calibrating..."

        # Regular Detection Logic
        left_ear = detector._calculate_ear(face_landmarks.landmark, detector.LEFT_EYE, iw, ih)
        right_ear = detector._calculate_ear(face_landmarks.landmark, detector.RIGHT_EYE, iw, ih)
        avg_ear = (left_ear + right_ear) / 2.0
        mar = detector._calculate_mar(face_landmarks.landmark, iw, ih)
        
        detector.ear_buffer.append(avg_ear)
        detector.mar_buffer.append(mar)
        smoothed_ear = sum(detector.ear_buffer) / len(detector.ear_buffer)
        smoothed_mar = sum(detector.mar_buffer) / len(detector.mar_buffer)
        
        is_distracted = detector._check_distraction(face_landmarks.landmark, iw, ih)
        
        # CNN Logic (simplified for single frame)
        probability = 1.0
        if detector.use_model:
            # We skip the complex cropping from detector.py for brevity, or implement it if needed
            # For the demo, EAR is usually enough, but let's try to keep it consistent
            is_drowsy_ear = (smoothed_ear < detector.EAR_THRESHOLD)
            is_drowsy = is_drowsy_ear
        else:
            is_drowsy = (smoothed_ear < detector.EAR_THRESHOLD)

        # Update counters
        if is_distracted:
            detector.consecutive_distracted_frames += 1
        else:
            detector.consecutive_distracted_frames = max(0, detector.consecutive_distracted_frames - 2)
            
        if is_drowsy:
            detector.consecutive_drowsy_frames += 1
        else:
            detector.consecutive_drowsy_frames = max(0, detector.consecutive_drowsy_frames - 2)

        # Determine Status
        if detector.consecutive_drowsy_frames >= detector.drowsy_threshold:
            status = "DROWSY"
            color = (0, 0, 255)
            is_alert_triggered = True
        elif detector.consecutive_distracted_frames >= detector.distraction_threshold:
            status = "DISTRACTED"
            color = (0, 165, 255)
            is_alert_triggered = True
        elif smoothed_mar > detector.MAR_THRESHOLD:
            status = "YAWNING"
            color = (0, 255, 255)
        
        # Draw UI on frame
        cv2.rectangle(frame_bgr, (0, 0), (iw, ih), color, 8)
        cv2.putText(frame_bgr, f"STATUS: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame_bgr, f"EAR: {smoothed_ear:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if is_alert_triggered:
            cv2.putText(frame_bgr, "!!! ALERT !!!", (iw//4, ih//2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5)

    else:
        status = "No face detected"

    # Convert back to RGB for Gradio
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), status

# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚗 AlertDrive AI")
    gr.Markdown("### Real-time Drowsiness & Distraction Detection System")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(sources=["webcam"], streaming=True, label="Live Webcam")
        with gr.Column():
            output_img = gr.Image(label="Detection Feed")
            status_text = gr.Textbox(label="Current Driver Status")
            
    input_img.stream(fn=predict, inputs=input_img, outputs=[output_img, status_text])
    
    gr.Markdown("---")
    gr.Markdown("Powered by BTEHub AI Solutions | TensorFlow | MediaPipe")

if __name__ == "__main__":
    demo.launch()
