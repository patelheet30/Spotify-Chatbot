import asyncio
import logging
import os

from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker

from app.api.spotify import SpotifyAPI
from app.chatbot.brain.handler import MusicInfoChatbot
from app.database.database import engine

load_dotenv()

logging.basicConfig(level=logging.INFO)


async def test_chatbot():
    """Test the MusicInfoChatbot with sample queries."""
    client_id = os.getenv("SPOTIPY_CLIENT_ID")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET")

    if not client_id or not client_secret:
        print("Error: Spotify API credentials not found in environment variables.")
        print("Please set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env file")
        return

    Session = sessionmaker(bind=engine)
    db_session = Session()

    spotify_api = SpotifyAPI(client_id=client_id, client_secret=client_secret)

    chatbot = MusicInfoChatbot(db_session=db_session, spotify_api=spotify_api)

    print("\n=== Music Information Chatbot ===")
    print("Type 'exit' to quit")

    while True:
        user_input = input("\n📱 You: ").strip()

        if user_input.lower() in ["exit", "quit", "bye"]:
            print("🤖 Bot: Goodbye!")
            break

        response, _ = await chatbot.get_response(user_input)
        print(f"🤖 Bot: {response}")

    db_session.close()


if __name__ == "__main__":
    asyncio.run(test_chatbot())
