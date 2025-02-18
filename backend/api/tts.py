# backend/api/tts.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.tts.tts_service import TTSService
from typing import Optional

router = APIRouter()
tts_service = TTSService()

class TTSRequest(BaseModel):
    text: str
    output_filename: Optional[str] = None

class TTSResponse(BaseModel):
    status: str
    file_path: Optional[str] = None
    text: Optional[str] = None
    file_size: Optional[int] = None
    error: Optional[str] = None

@router.post("/generate")
async def generate_speech(request: TTSRequest):
    """Generate speech from text."""
    try:
        result = await tts_service.generate_speech(
            text=request.text,
            output_filename=request.output_filename
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))