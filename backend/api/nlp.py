from fastapi import APIRouter, HTTPException
from core.nlp.mistral_nlp import MistralNLP

router = APIRouter()
nlp_model = MistralNLP()

@router.post("/generate-response")
async def generate_response(user_input: str):
    """API to process user input and generate AI response."""
    try:
        ai_response = nlp_model.generate_text(user_input)
        return {"response": ai_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
