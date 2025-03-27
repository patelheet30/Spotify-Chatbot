import os
from typing import Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.spotify import SpotifyAPI
from app.chatbot.brain.handler import MusicInfoChatbot
from app.chatbot.voice.utils import get_temp_audio_file
from app.database import crud
from app.database.database import get_db, init_db

load_dotenv()

app = FastAPI(title="Spotify Chatbot")


def get_chatbot(db: Session = Depends(get_db)) -> MusicInfoChatbot:
    global chatbot_instance
    if chatbot_instance is None:
        chatbot_instance = MusicInfoChatbot(db_session=db, spotify_api=spotify_api)
    return chatbot_instance


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

spotify_api = SpotifyAPI(
    client_id=os.getenv("SPOTIPY_CLIENT_ID", ""),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET", ""),
)

chatbot_instance = None

os.makedirs("static", exist_ok=True)
os.makedirs("static/uploads", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_event():
    init_db()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    context: Dict
    voice_enabled: Optional[bool] = False
    audio_response: Optional[str] = None


class ImageClassificationRequest(BaseModel):
    image_url: Optional[str] = None
    use_ensemble: bool = False


class ImageClassificationResponse(BaseModel):
    era: str
    confidence: str
    model: str


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        html_content = f.read()
    return html_content


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, chatbot: MusicInfoChatbot = Depends(get_chatbot)):
    response, context = await chatbot.get_response(request.message)

    simplified_context = {}

    if "audio_response" in context:
        simplified_context["audio_response"] = context["audio_response"]

    if chatbot.voice_enabled and "audio_response" not in simplified_context:
        audio_path = get_temp_audio_file()
        if chatbot.text_to_speech.save_to_file(response, audio_path):
            simplified_context["audio_response"] = audio_path

    if "query_type" in context:
        simplified_context["query_type"] = context["query_type"]

    return ChatResponse(
        response=response,
        context=simplified_context,
        voice_enabled=chatbot.voice_enabled,
        audio_response=simplified_context.get("audio_response"),
    )


@app.post("/api/upload-image", response_model=ImageClassificationResponse)
async def upload_image(
    file: UploadFile = File(...),
    use_ensemble: bool = Form(False),
    chatbot: MusicInfoChatbot = Depends(get_chatbot),
):
    file_path = f"static/uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    result = await chatbot.classify_album_era(file_path, use_ensemble)

    return ImageClassificationResponse(
        era=result["era"], confidence=result["confidence"], model=result["model"]
    )


@app.post("/api/classify-image-url", response_model=ImageClassificationResponse)
async def classify_image_url(
    request: ImageClassificationRequest,
    chatbot: MusicInfoChatbot = Depends(get_chatbot),
):
    if not request.image_url:
        return {"error": "Image URL is required"}

    result = await chatbot.classify_album_era(request.image_url, request.use_ensemble)

    if "era" not in result or "confidence" not in result or "model" not in result:
        return ImageClassificationResponse(
            era="Error", confidence="0%", model="Unknown"
        )

    return ImageClassificationResponse(
        era=result["era"], confidence=result["confidence"], model=result["model"]
    )


@app.post("/api/toggle-voice")
async def toggle_voice(
    enable: bool = Form(...), chatbot: MusicInfoChatbot = Depends(get_chatbot)
):
    message, context = chatbot.toggle_voice(enable)
    return {"message": message, "voice_enabled": context.get("voice_enabled", False)}


@app.post("/api/listen-voice")
async def listen_voice(chatbot: MusicInfoChatbot = Depends(get_chatbot)):
    response, context = await chatbot.listen_voice()

    simplified_context = {}

    if "audio_response" in context:
        simplified_context["audio_response"] = context["audio_response"]

    if chatbot.voice_enabled and "audio_response" not in simplified_context:
        response_parts = response.split("My Response: ")
        if len(response_parts) > 1:
            bot_response = response_parts[1]
            audio_path = get_temp_audio_file()
            success = chatbot.text_to_speech.save_to_file(bot_response, audio_path)
            if success:
                simplified_context["audio_response"] = audio_path

    return {"response": response, "context": simplified_context}


@app.get("/api/check-voice-available")
async def check_voice_available(chatbot: MusicInfoChatbot = Depends(get_chatbot)):
    return {"available": chatbot.voice_available}


@app.get("/api/artists", response_model=List[Dict])
async def get_artists(db: Session = Depends(get_db), skip: int = 0, limit: int = 100):
    artists = crud.artist.get_multi(db, skip=skip, limit=limit)
    return [
        {
            "id": artist.id,
            "name": artist.name,
            "popularity": artist.popularity,
            "followers": artist.followers,
            "image_url": artist.image_url,
        }
        for artist in artists
    ]


@app.get("/api/albums", response_model=List[Dict])
async def get_albums(db: Session = Depends(get_db), skip: int = 0, limit: int = 100):
    albums = crud.album.get_multi(db, skip=skip, limit=limit)
    return [
        {
            "id": album.id,
            "name": album.name,
            "album_type": album.album_type,
            "total_tracks": album.total_tracks,
            "release_date": album.release_date,
            "image_url": album.image_url,
        }
        for album in albums
    ]


@app.get("/api/tracks", response_model=List[Dict])
async def get_tracks(db: Session = Depends(get_db), skip: int = 0, limit: int = 100):
    tracks = crud.track.get_multi(db, skip=skip, limit=limit)
    return [
        {
            "id": track.id,
            "name": track.name,
            "duration_ms": track.duration_ms,
            "popularity": track.popularity,
            "preview_url": track.preview_url,
        }
        for track in tracks
    ]


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
