import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv

from app.api.spotify import SpotifyAPI
from app.database.database import SessionLocal


async def test_spotify_integration():
    """Test the complete Spotify data integration pipeline."""

    # Load environment variables
    load_dotenv()

    # Initialize Spotify integration
    spotify = SpotifyAPI(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),  # type: ignore
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),  # type: ignore
    )

    # Read artist IDs from file
    with open("data/artist_ids.txt", "r") as f:
        # Split on '-' and strip whitespace to get clean IDs
        artist_ids = [line.split("-")[0].strip() for line in f.readlines()]

    db = SessionLocal()
    start_time = datetime.now()

    try:
        print(f"\nStarting data collection at {start_time.strftime('%H:%M:%S')}")
        print("=" * 50)

        for i, artist_id in enumerate(artist_ids[0:], 1):
            try:
                print(f"\nProcessing artist {i}/{len(artist_ids)}: {artist_id}")
                print("-" * 30)

                # Fetch complete artist data (this will trigger album and track collection)
                print("Fetching artist data...")
                artist = await spotify.get_artist_data(artist_id, db)
                print(f"✓ Artist: {artist.name}")
                print(f"  - Followers: {artist.followers:,}")
                print(f"  - Popularity: {artist.popularity}")
                print(f"  - Genres: {', '.join(genre.name for genre in artist.genres)}")

                # Show album data
                albums = await spotify.get_artist_albums(artist_id, db)
                print(f"✓ Albums: {len(albums)}")
                for album in albums:
                    print(f"  - {album.name} ({album.album_type})")

                    # Get tracks for each album
                    tracks = await spotify.get_album_tracks(album.spotify_id, db)  # type: ignore
                    print(f"    → {len(tracks)} tracks")

                    # Count tracks with audio features
                    tracks_with_features = sum(
                        1 for track in tracks if track.audio_features is not None
                    )
                    print(f"    → {tracks_with_features} tracks with audio features")

                print(f"✓ Successfully processed {artist.name}")

            except Exception as e:
                print(f"❌ Error processing artist {artist_id}: {str(e)}")
                continue

            # Add a visual separator between artists
            print("\n" + "=" * 50)

    finally:
        end_time = datetime.now()
        duration = end_time - start_time
        print(f"\nData collection completed at {end_time.strftime('%H:%M:%S')}")
        print(f"Total duration: {duration}")
        db.close()


if __name__ == "__main__":
    asyncio.run(test_spotify_integration())
