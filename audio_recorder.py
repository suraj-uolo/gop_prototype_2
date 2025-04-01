import os
import wave
import logging
import sounddevice as sd
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AudioRecorder:
    def __init__(self, recordings_dir="recordings", samplerate=16000):
        self.recordings_dir = recordings_dir
        self.samplerate = samplerate
        self.recorded_audio = None
        self.filepath = None
        os.makedirs(self.recordings_dir, exist_ok=True)

    def record_audio(self, duration: int = 5, filename: str = None) -> str:
        """Records audio for a given duration and saves it as a WAV file."""
        if not filename:
            filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        self.filepath = os.path.join(self.recordings_dir, filename)

        try:
            logging.info(f"🎙️ Recording for {duration} seconds...")

            # Correct way to record audio
            self.recorded_audio = sd.rec(int(self.samplerate * duration), samplerate=self.samplerate, channels=1, dtype=np.int16)
            sd.wait()  # Wait for recording to finish
            # Save as WAV
            return self.save_audio()

        except sd.PortAudioError as e:
            logging.error(f"🎤 Sounddevice error: {e}")
            raise
        except Exception as e:
            logging.error(f"❌ Unexpected error while recording: {e}")
            raise

    def save_audio(self):
        with wave.open(self.filepath, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio (2 bytes per sample)
            wf.setframerate(self.samplerate)
            wf.writeframes(self.recorded_audio.tobytes())
            logging.info(f"✅ Recording saved: {self.filepath}")
        return self.filepath