import asyncio
import os

from dotenv import load_dotenv
from sqlalchemy.orm import Session

from app.api.spotify import SpotifyAPI
from app.database.database import SessionLocal


async def debug_spotify():
    load_dotenv()

    spotify = SpotifyAPI(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
    )

    db = SessionLocal()
    try:
        # Test with The Weeknd's ID
        artist_id = "1Xyo4u8uXC1ZmMpatF05PJ"

        # First, let's see what raw data we get from Spotify
        artist_albums = spotify.spotify.artist_albums(artist_id)
        print("\nRaw album data from Spotify:")
        first_album = artist_albums["items"][0]
        print(f"First album keys: {first_album.keys()}")
        print(f"First album data: {first_album}")

        # Now let's check what happens when we try to get full album data
        full_album = spotify.spotify.album(first_album["id"])
        print("\nFull album data:")
        print(f"Full album keys: {full_album.keys()}")
        print(f"Popularity present: {'popularity' in full_album}")
        print(f"Popularity value: {full_album.get('popularity')}")

    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(debug_spotify())
