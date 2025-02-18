# backend/core/tts/tts_service.py

import pyttsx3
from pathlib import Path
import os
import logging
from typing import Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTSService:
    """Text-to-Speech service using pyttsx3."""
    
    def __init__(self):
        self._output_dir = Path("backend/core/tts/audio_cache")
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self.engine = None
        self.initialize_tts()

    def initialize_tts(self) -> None:
        try:
            logger.info("Initializing TTS engine")
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            
            voices = self.engine.getProperty('voices')
            for voice in voices:
                if "english" in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
                    
            logger.info("TTS initialization successful")
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {str(e)}")
            raise

    async def generate_speech(self, text: str, output_filename: Optional[str] = None) -> dict:
        try:
            if not output_filename:
                output_filename = f"speech_{hash(text)}.wav"
            
            output_path = self._output_dir / output_filename

            self.engine.save_to_file(text, str(output_path))
            self.engine.runAndWait()

            return {
                "status": "success",
                "file_path": str(output_path),
                "text": text,
                "file_size": os.path.getsize(output_path)
            }

        except Exception as e:
            logger.error(f"Speech generation failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def cleanup_old_files(self, max_age_hours: int = 24) -> None:
        try:
            current_time = time.time()
            for file_path in self._output_dir.glob("*.wav"):
                if (current_time - os.path.getctime(file_path)) > (max_age_hours * 3600):
                    os.remove(file_path)
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")