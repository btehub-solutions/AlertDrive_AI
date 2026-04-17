import pygame
import random
import os

class AlertSystem:
    """Manages alerts, session tracking, and reporting for drowsiness detection."""

    def __init__(self):
        """Initialize the alert system and audio mixer."""
        # Initialize pygame mixer for audio
        try:
            pygame.mixer.init()
            self.audio_enabled = True
        except pygame.error as e:
            print(f"Warning: Could not initialize audio mixer ({e}). Sound alerts will be disabled.")
            self.audio_enabled = False

        self.alert_sound_path = os.path.join('alerts', 'alert.wav')
        
        if self.audio_enabled and os.path.exists(self.alert_sound_path):
            self.alert_sound = pygame.mixer.Sound(self.alert_sound_path)
        else:
            self.alert_sound = None
            print(f"Warning: {self.alert_sound_path} not found. Please add alert.wav to the alerts folder.")

        # Warning messages for DROWSY state
        self.drowsy_messages = [
            "DROWSINESS DETECTED - Please stay alert",
            "You are getting drowsy - Pull over and rest",
            "DANGER - Take a short break immediately",
            "Eyes closing detected - Stop driving now"
        ]
        
        # Confirmation message for NATURAL state
        self.natural_message = "Driver is Alert - Safe to drive"

        # Session tracking statistics
        self.total_frames = 0
        self.drowsy_frames = 0
        self.alert_frames = 0

    def play_alert(self):
        """Play an audio alert sound using pygame mixer."""
        if self.audio_enabled and self.alert_sound:
            if not pygame.mixer.get_busy():
                self.alert_sound.play()
        else:
            # Fallback if no sound file
            pass # Removed system beep to avoid logs

    def stop_alert(self):
        """Stop the currently playing alert sound."""
        if self.audio_enabled and pygame.mixer.get_busy():
            pygame.mixer.stop()

    def get_display_message(self, status):
        """Return the appropriate warning or confirmation message."""
        if status == "DROWSY":
            return random.choice(self.drowsy_messages)
        else:
            return self.natural_message

    def process_frame(self, status, is_alert_triggered):
        """Track statistics and manage sound for a single frame."""
        self.total_frames += 1
        
        if status == "DROWSY":
            self.drowsy_frames += 1
            
        if is_alert_triggered:
            self.alert_frames += 1
            self.play_alert()
        else:
            self.stop_alert()

    def generate_report(self):
        """Calculate final risk level and generate a session summary report."""
        drowsiness_rate = 0
        if self.total_frames > 0:
            drowsiness_rate = (self.drowsy_frames / self.total_frames) * 100

        # Calculate final risk level
        if drowsiness_rate > 30:
            risk_level = "HIGH RISK"
        elif drowsiness_rate > 10:
            risk_level = "MODERATE RISK"
        else:
            risk_level = "LOW RISK"

        # Generate the formatted report
        report = [
            "\n" + "="*40,
            "SESSION SUMMARY REPORT",
            "="*40,
            f"Total frames analyzed: {self.total_frames}",
            f"Drowsy frame count: {self.drowsy_frames}",
            f"Alert frame count: {self.alert_frames}",
            f"Overall drowsiness rate: {drowsiness_rate:.2f}%",
            "-"*40,
            f"FINAL RISK LEVEL: {risk_level}",
            "="*40 + "\n"
        ]
        
        return "\n".join(report)
