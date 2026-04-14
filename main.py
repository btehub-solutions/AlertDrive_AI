import json
import os
import sys
import cv2
from datetime import datetime

# Import detector and alert_system modules
import alert_system
from detector import DrowsinessDetector


def display_welcome_screen():
    """Displays the welcome screen showing version and credits."""
    print("========================================")
    print("  AlertDrive AI v1.0")
    print("  Powered by TensorFlow and MobileNetV2")
    print("  Developer: BTEHub Team")
    print("========================================")
    print()


def save_report_to_json(alert_sys):
    """Saves the final session summary report into a JSON file with a timestamp."""
    timestamp_iso = datetime.now().isoformat()
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    total = alert_sys.total_frames
    drowsy = alert_sys.drowsy_frames
    alerts = alert_sys.alert_frames
    
    drowsiness_rate = 0
    if total > 0:
        drowsiness_rate = (drowsy / total) * 100

    if drowsiness_rate > 30:
        risk_level = "HIGH RISK"
    elif drowsiness_rate > 10:
        risk_level = "MODERATE RISK"
    else:
        risk_level = "LOW RISK"
        
    report_data = {
        "session_timestamp": timestamp_iso,
        "metrics": {
            "total_frames": total,
            "drowsy_frames": drowsy,
            "alert_frames": alerts,
            "drowsiness_rate_percent": round(drowsiness_rate, 2),
            "final_risk_level": risk_level
        }
    }
    
    filename = f"session_report_{timestamp_file}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=4)
        print(f"[INFO] Session report saved to {filename}")
    except Exception as e:
        print(f"[ERROR] Failed to save session report: {e}")


def main():
    display_welcome_screen()
    
    # Ask user to press ENTER to start detection
    try:
        input("Press ENTER to start detection...")
    except KeyboardInterrupt:
        print("\nExiting AlertDrive AI...")
        sys.exit(0)

        
    detector = None
    try:
        # Start the webcam detection loop (integrate real time alerts is handled in detector.run)
        detector = DrowsinessDetector()
        detector.run()
        
    except KeyboardInterrupt:
        # Handle keyboard interrupt gracefully during detection
        print("\n[INFO] Keyboard interrupt detected. Stopping detection gracefully...")
        
        # Ensure windows are destroyed if interrupted mid-loop
        cv2.destroyAllWindows()
        
        if detector and hasattr(detector, 'alert_system'):
            # Manually stop alerts and print the summary since detector.run() was interrupted
            detector.alert_system.stop_alert()
            # On exit display full session summary report
            print(detector.alert_system.generate_report())
            
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        
    finally:
        # Save session report to a JSON file with timestamp
        if detector and hasattr(detector, 'alert_system'):
            save_report_to_json(detector.alert_system)


if __name__ == "__main__":
    main()
