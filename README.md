---
title: AlertDrive AI
emoji: 🚗
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
python_version: 3.11
---

# AlertDrive AI: Advanced Driver Monitoring System (DMS) 🚗💤

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-green.svg)](https://mediapipe.dev/)

**AlertDrive AI** is a professional-grade safety application designed to prevent road accidents caused by driver fatigue and distraction. By leveraging state-of-the-art Computer Vision and Deep Learning, the system monitors driver behavior in real-time, providing instant life-saving audio alerts when danger is detected.

## ✨ Key Features

*   **🔍 Multi-Factor Detection:** Tracks Eye Aspect Ratio (EAR) for drowsiness, Mouth Aspect Ratio (MAR) for yawning, and Head Pose (Pitch/Yaw) for distraction.
*   **🤖 Hybrid AI Architecture:** Combines geometric computer vision with a specialized **MobileNetV2 CNN model** for maximum accuracy and reduced false positives.
*   **⚙️ Intelligent Auto-Calibration:** Automatically learns the driver's unique "open-eye" baseline during the first 3 seconds of operation to adapt to different users.
*   **🌓 Adaptive Processing:** Utilizes **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to ensure reliable detection under varying car lighting conditions (e.g., night driving or tunnels).
*   **📊 Safety Analytics:** Generates detailed JSON session reports and "Risk Level" assessments (Low, Moderate, High) after every trip.
*   **🔊 Real-time Audio Alerts:** Integrated pygame-based sound system that triggers "Wake Up" or "Eyes on Road" warnings immediately.

## 🛠️ Tech Stack

- **Core:** Python 3.12+
- **Deep Learning:** TensorFlow & Keras (MobileNetV2)
- **Vision:** MediaPipe (Face Landmarker Tasks API), OpenCV
- **Audio:** Pygame
- **Processing:** NumPy, SciPy

## 🚀 Getting Started

### Prerequisites
- A webcam
- Python 3.12 or higher

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/btehub-solutions/AlertDrive_AI.git
   cd AlertDrive_AI
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure models are present:**
   Place your `alertdrive_best_model.h5` and `face_landmarker.task` in the `models/` directory.

### Running the App
```bash
python main.py
```

## 📈 How It Works

1.  **Calibration:** On startup, the system asks you to look forward to establish your baseline EAR.
2.  **Monitoring:** The logic evaluates:
    - **EAR < Threshold:** Triggers Drowsiness Alert.
    - **MAR > Threshold:** Detects Yawning (Early Warning).
    - **Pitch/Yaw Offset:** Triggers Distraction Alert (Eyes on Road).
3.  **Reporting:** Upon exit, a `session_report_YYYYMMDD_HHMMSS.json` is generated with your safety metrics.

## 👨‍💻 Developer
Developed with ❤️ by the **BTEHub Team**.
