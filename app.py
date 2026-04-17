import gradio as gr
import cv2
import numpy as np
import os
from detector import DrowsinessDetector

# Prevent TensorFlow from grabbing all memory and conflicting with MediaPipe
# Prevent ALSA/Pygame errors and library conflicts
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

# Initialize the detector
detector = DrowsinessDetector(model_path='models/alertdrive_best_model.h5')
is_processing = False # Global flag to prevent overlapping frames

def predict(frame):
    global is_processing
    
    if frame is None:
        return None, "Waiting for camera..."

    if is_processing:
        return None, gr.skip()

    try:
        is_processing = True
        
        # Handle new Gradio format if necessary
        if isinstance(frame, dict): 
            frame = frame["composite"]
            
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        ih, iw, _ = frame_bgr.shape
        
        enhanced_frame = detector._apply_clahe(frame_bgr)
        rgb_frame = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB)
        
        import mediapipe as mp
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.face_landmarker.detect(mp_image)
        
        status = "NATURAL"
        color = (0, 255, 0)
        is_alert_triggered = False
        
        class MockLandmarks:
            def __init__(self, lmks):
                self.landmark = lmks

        if results.face_landmarks:
            face_landmarks = MockLandmarks(results.face_landmarks[0])
            
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
                
                is_processing = False
                return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), f"Calibrating: {progress}%"

            left_ear = detector._calculate_ear(face_landmarks.landmark, detector.LEFT_EYE, iw, ih)
            right_ear = detector._calculate_ear(face_landmarks.landmark, detector.RIGHT_EYE, iw, ih)
            avg_ear = (left_ear + right_ear) / 2.0
            mar = detector._calculate_mar(face_landmarks.landmark, iw, ih)
            
            detector.ear_buffer.append(avg_ear)
            detector.mar_buffer.append(mar)
            smoothed_ear = sum(detector.ear_buffer) / len(detector.ear_buffer)
            smoothed_mar = sum(detector.mar_buffer) / len(detector.mar_buffer)
            
            is_distracted = detector._check_distraction(face_landmarks.landmark, iw, ih)
            
            # Simple Drowsy Logic
            is_drowsy = (smoothed_ear < detector.EAR_THRESHOLD) if not is_distracted else False

            if is_distracted: detector.consecutive_distracted_frames += 1
            else: detector.consecutive_distracted_frames = max(0, detector.consecutive_distracted_frames - 2)
                
            if is_drowsy: detector.consecutive_drowsy_frames += 1
            else: detector.consecutive_drowsy_frames = max(0, detector.consecutive_drowsy_frames - 2)

            if detector.consecutive_drowsy_frames >= detector.drowsy_threshold:
                status, color, is_alert_triggered = "DROWSY", (0, 0, 255), True
            elif detector.consecutive_distracted_frames >= detector.distraction_threshold:
                status, color, is_alert_triggered = "DISTRACTED", (0, 165, 255), True
            elif smoothed_mar > detector.MAR_THRESHOLD:
                status, color = "YAWNING", (0, 255, 255)
            
            cv2.rectangle(frame_bgr, (0, 0), (iw, ih), color, 8)
            cv2.putText(frame_bgr, f"STATUS: {status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame_bgr, f"EAR: {smoothed_ear:.2f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            if is_alert_triggered:
                cv2.putText(frame_bgr, "!!! ALERT !!!", (iw//4, ih//2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5)
        else:
            status = "No face detected"

        is_processing = False
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), status

    except Exception as e:
        is_processing = False
        return None, f"AI Error: {str(e)}"

# Create Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚗 AlertDrive AI")
    gr.Markdown("### Real-time Drowsiness & Distraction Detection System")
    
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(sources=["webcam"], streaming=True, label="Live Webcam")
            with gr.Row():
                start_btn = gr.Button("▶️ Start AI Tracking", variant="primary")
                stop_btn = gr.Button("⏹️ Stop")
        with gr.Column():
            output_img = gr.Image(label="AI Detection Output")
            status_text = gr.Textbox(label="Driver State Feedback")
            telemetry = gr.Markdown("Check if frames are processing...")
            
    # Trigger the stream on click for better browser compatibility
    stream_event = start_btn.click(
        fn=predict, 
        inputs=input_img, 
        outputs=[output_img, status_text],
        show_progress="hidden"
    )
    
    # We also keep the stream event for continuous tracking
    input_img.stream(
        fn=predict, 
        inputs=input_img, 
        outputs=[output_img, status_text],
        show_progress="hidden"
    )
    
    stop_btn.click(fn=None, cancels=[stream_event])
    
    gr.Markdown("---")
    gr.Markdown("Powered by BTEHub AI Solutions | TensorFlow | MediaPipe")

if __name__ == "__main__":
    # On Hugging Face Spaces, server_name="0.0.0.0" and server_port=7860 are essential.
    # Disabling ssr_mode and explicitly setting share=False to avoid connectivity checks.
    demo.launch()
