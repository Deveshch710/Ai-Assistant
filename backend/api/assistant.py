# backend/api/assistant.py

from fastapi import APIRouter, HTTPException
from core.stt.speech_recognition import SpeechToText
from core.tts.tts_service import TTSService
from pydantic import BaseModel
from typing import Optional

router = APIRouter()
stt = SpeechToText(model="base", use_vosk=False)  # Use tiny model by default
tts = TTSService()

class AssistantResponse(BaseModel):
    text: str
    audio_file: Optional[str] = None

@router.get("/speech-to-text")
async def speech_to_text():
    try:
        text = stt.recognize_speech()
        if text.startswith("Error"):
            raise HTTPException(status_code=500, detail=text)
        return {"transcribed_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/respond")
async def respond_with_voice(text: str):
    """Respond to user with both text and voice."""
    try:
        # Generate speech from text
        speech_result = await tts.generate_speech(text)
        
        if speech_result["status"] == "error":
            raise HTTPException(status_code=500, detail=speech_result["error"])
            
        return AssistantResponse(
            text=text,
            audio_file=speech_result["file_path"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))