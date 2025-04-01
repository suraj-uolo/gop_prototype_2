import os
import logging
import noisereduce as nr
import numpy as np
from scipy.io import wavfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class AudioCleaner:
    def __init__(self, cleaned_dir="cleaned_recordings"):
        """Initialize the cleaner with a directory to save cleaned audio."""
        self.cleaned_dir = cleaned_dir
        os.makedirs(self.cleaned_dir, exist_ok=True)

    def clean_audio(self, audio_path: str, use_noise_sample=False) -> str:
        """
        Cleans the given recorded audio by reducing noise.
        
        Args:
            audio_path (str): Path to the recorded audio file.
            use_noise_sample (bool): Whether to use a noise sample for better cleaning.
        
        Returns:
            str: Path to the cleaned audio file.
        """
        try:
            logging.info(f"🔄 Loading audio: {audio_path}")
            rate, data = wavfile.read(audio_path)

            if len(data.shape) == 2:  
                logging.info("🎵 Stereo detected, converting to mono.")
                data = np.mean(data, axis=1).astype(np.int16)  

            # Noise Reduction
            if use_noise_sample:
                logging.info("🎚️ Using first 5000 samples as noise profile.")
                noise_sample = data[:5000]  # Select an initial segment as noise
                cleaned_data = nr.reduce_noise(y=data, sr=rate, y_noise=noise_sample)
            else:
                logging.info("📉 Applying non-stationary noise reduction.")
                cleaned_data = nr.reduce_noise(y=data, sr=rate, stationary=False, prop_decrease=0.9)

            # Save cleaned file
            cleaned_filename = f"cleaned_{os.path.basename(audio_path)}"
            cleaned_path = os.path.join(self.cleaned_dir, cleaned_filename)
            wavfile.write(cleaned_path, rate, cleaned_data.astype(np.int16))

            logging.info(f"✅ Cleaned audio saved: {cleaned_path}")
            return cleaned_path

        except Exception as e:
            logging.error(f"❌ Error cleaning audio: {e}")
            raise
