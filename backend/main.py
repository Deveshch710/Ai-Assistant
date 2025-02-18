from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import assistant, user, nlp, tts  # Import the new NLP API

app = FastAPI(title="AI Assistant", version="1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routes
app.include_router(assistant.router, prefix="/assistant", tags=["Assistant"])
app.include_router(user.router, prefix="/user", tags=["User"])
app.include_router(nlp.router, prefix="/nlp", tags=["NLP"])  # Add NLP API
app.include_router(tts.router, prefix="/tts", tags=["TTS"])  # Add TTS API

@app.get("/")
def home():
    return {"message": "AI Assistant API is Running!"}

if __name__ == "__main__":
    import uvicorn
    print("Starting server at http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)