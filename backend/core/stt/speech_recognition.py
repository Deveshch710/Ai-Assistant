# backend/core/stt/speech_recognition.py

# we are using which is a wrapper around the Whisper library for speech recognition.
#This library provides a simple interface to record audio, enhance
#its quality, and transcribe it using Whisper's speech recognition models.
#We are using its base model as the model increase it will require more memory and GPU. But give more better output more presise, and tewer samlple rate which give the best output
#We are also using the Vosk library as an alternative for speech recognition.
#we are recoring the audio, then elementaing the noice by doing the noise reduction using spectral gating, then we are applying the bandpass filter to focus on speech frequencies (300Hz - 3000Hz).
#Then we are transcribing the audio with improved accuracy by using Whisper library.
#We are also using the Vosk library as an alternative for speech recognition.




import whisper
import vosk
import sounddevice as sd
import queue
import json
import numpy as np
import torch
from scipy import signal

class SpeechToText:
    def __init__(self, model="base", use_vosk=False):
        self.use_vosk = use_vosk
        if self.use_vosk:
            self.model = vosk.Model("model")
            self.recognizer = vosk.KaldiRecognizer(self.model, 16000)
        else:
            # Try to use a better model with memory management
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                # Try to load base model first for better accuracy
                self.model = whisper.load_model(model, device=self.device)
            except RuntimeError:
                print("Memory error with base model, falling back to tiny model")
                self.device = "cpu"
                self.model = whisper.load_model("tiny", device=self.device)

    def enhance_audio(self, audio_data, samplerate=16000):
        """Enhance audio quality for better recognition"""
        # Normalize audio
        audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Apply noise reduction using spectral gating
        # Calculate noise profile from the first 1000 samples
        noise_sample = audio_data[:1000]
        noise_profile = np.mean(np.abs(noise_sample))
        
        # Apply simple noise gate
        gate_threshold = noise_profile * 2
        audio_data[np.abs(audio_data) < gate_threshold] = 0
        
        # Apply bandpass filter to focus on speech frequencies (300Hz - 3000Hz)
        nyquist = samplerate / 2
        low = 300 / nyquist
        high = 3000 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        audio_data = signal.filtfilt(b, a, audio_data)
        
        return audio_data

    def record_audio(self, duration=5, samplerate=16000):
        """Records audio with improved quality"""
        # Pre-allocate with higher quality settings
        total_samples = int(samplerate * duration)
        audio_data = np.zeros((total_samples, 1), dtype=np.float32)
        samples_recorded = 0
        
        def callback(indata, frames, time, status):
            nonlocal samples_recorded
            if status:
                print(f"Warning: {status}")
            end_idx = min(samples_recorded + len(indata), total_samples)
            if samples_recorded < total_samples:
                audio_data[samples_recorded:end_idx] = indata[:end_idx-samples_recorded]
                samples_recorded += len(indata)

        try:
            # Use higher quality recording settings
            with sd.InputStream(samplerate=samplerate, 
                              channels=1,
                              callback=callback,
                              blocksize=1024,
                              device=None,  # Use default device
                              latency='low',  # Lower latency for better quality
                              dtype=np.float32):
                print(f"Recording for {duration} seconds...")
                sd.sleep(int(duration * 1000))
        except Exception as e:
            print(f"Error recording audio: {e}")
            return None

        recorded_audio = audio_data.flatten()
        # Enhance the recorded audio
        enhanced_audio = self.enhance_audio(recorded_audio, samplerate)
        return enhanced_audio

    def transcribe_audio(self, audio):
        """Transcribes audio with improved accuracy"""
        if audio is None:
            return "Error recording audio"

        try:
            if self.use_vosk:
                self.recognizer.AcceptWaveform(audio.tobytes())
                result = json.loads(self.recognizer.Result())
                return result.get("text", "")

            # Prepare audio for Whisper
            audio = audio.astype(np.float32)
            
            # Ensure proper audio levels
            if np.abs(audio).max() > 0:
                audio = audio / np.abs(audio).max()

            # Configure Whisper for better accuracy
            options = {
                "language": "en",
                "task": "transcribe",
                "fp16": False,
                "beam_size": 5,  # Increase beam size for better accuracy
                "best_of": 5,    # Consider more candidates
                "patience": 2,    # Increase patience for better results
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
            }

            with torch.inference_mode():
                result = self.model.transcribe(
                    audio,
                    **options
                )
                
                # Post-process the text
                text = result["text"].strip()
                # Remove multiple spaces
                text = ' '.join(text.split())
                return text

        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return f"Error transcribing audio: {str(e)}"

    def recognize_speech(self, duration=5):
        """Captures and transcribes speech with improved accuracy"""
        try:
            audio = self.record_audio(duration)
            if audio is not None:
                # Multiple transcription attempts for better accuracy
                attempts = 2
                transcriptions = []
                
                for _ in range(attempts):
                    text = self.transcribe_audio(audio)
                    if not text.startswith("Error"):
                        transcriptions.append(text)
                
                if transcriptions:
                    # Return the longest transcription as it's likely the most complete
                    return max(transcriptions, key=len)
                else:
                    return "Could not transcribe audio clearly"
            
            return "Error: No audio recorded"
        except Exception as e:
            return f"Error in speech recognition: {str(e)}"