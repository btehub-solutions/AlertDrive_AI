import numpy as np
from scipy.io.wavfile import write
import os

def create_beep(frequency, duration, sample_rate=44100):
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate a sine wave
    # A multiplier of 32767 scales the float (-1 to 1) to the int16 range
    audio_data = np.int16(32767 * np.sin(2 * np.pi * frequency * t))
    return audio_data

if __name__ == "__main__":
    frequency = 440.0  # 440 Hz
    duration = 0.5     # 0.5 seconds
    sample_rate = 44100 # standard audio sample rate
    
    audio_data = create_beep(frequency, duration, sample_rate)
    
    # Ensure the target directory exists
    os.makedirs('alerts', exist_ok=True)
    
    output_path = os.path.join('alerts', 'alert.wav')
    write(output_path, sample_rate, audio_data)
    
    print(f"Successfully generated alert tone at {frequency}Hz for {duration}s -> {output_path}")
