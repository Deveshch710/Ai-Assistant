# backend/api/assistant.py

from fastapi import APIRouter, HTTPException
from core.stt.speech_recognition import SpeechToText

router = APIRouter()
stt = SpeechToText(model="base", use_vosk=False)  # Use tiny model by default

@router.get("/speech-to-text")
async def speech_to_text():
    try:
        text = stt.recognize_speech()
        if text.startswith("Error"):
            raise HTTPException(status_code=500, detail=text)
        return {"transcribed_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))