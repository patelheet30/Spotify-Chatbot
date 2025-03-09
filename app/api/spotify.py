from typing import Dict, List, Optional

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from sqlalchemy.orm import Session

from app.database import crud, models
from app.utils.cache import Cache


class SpotifyAPI:
    """Spotify API client with caching support."""

    def __init__(self, client_id: str, client_secret: str, cache_duration: int = 3600):
        self.spotify = spotipy.Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=client_id, client_secret=client_secret
            )
        )
        self.cache = Cache(expiration_time=cache_duration)

    async def get_artist_data(
        self, artist_id: str, db_session: Session
    ) -> models.Artist:
        """Get artist data from Spotify API.

        Args:
            artist_id (str): The Spotify ID of the artist.
            db_session (Session): SQLAlchemy session object

        Returns:
            models.Artist: The artist model instance.
        """
        cache_key = f"artist_{artist_id}"

        if cached_data := self.cache.get(cache_key):
            return crud.artist.get_by_spotify_id(
                db_session, artist_id
            ) or self._create_artist(db_session, cached_data)

        artist_data = self.spotify.artist(artist_id)
        self.cache.set(cache_key, artist_data)

        artist = crud.artist.get_by_spotify_id(
            db_session, artist_id
        ) or self._create_artist(
            db_session,
            artist_data,  # type: ignore
        )

        albums = await self.get_artist_albums(artist_id, db_session)

        for album in albums:
            await self.get_album_tracks(album.spotify_id, db_session)  # type: ignore

        return artist

    async def get_artist_albums(
        self, artist_id: str, db_session: Session
    ) -> List[models.Album]:
        """Get all albums by an artist from Spotify API.

        Args:
            artist_id (str): The artist ID
            db_session (Session): SQLAlchemy session object

        Returns:
            List[models.Album]: The list of album model instances.
        """
        cache_key = f"artist_albums_{artist_id}"

        if cached_data := self.cache.get(cache_key):
            albums = []
            for album_data in cached_data:
                album = crud.album.get_by_spotify_id(
                    db_session, album_data["id"]
                ) or self._create_album(db_session, album_data)
                albums.append(album)

            return albums

        albums_data = []
        for album_type in ["album", "single", "compilation"]:
            results = self.spotify.artist_albums(
                artist_id,
                album_type=album_type,
                limit=50,
            )

            albums_data.extend(results["items"])  # type: ignore

            while results["next"]:  # type: ignore
                results = self.spotify.next(results)
                albums_data.extend(results["items"])  # type: ignore

        self.cache.set(cache_key, albums_data)

        albums = []
        for album_data in albums_data:
            full_album_data = self.spotify.album(album_data["id"])
            album = crud.album.get_by_spotify_id(
                db_session, album_data["id"]
            ) or self._create_album(db_session, full_album_data)  # type: ignore
            albums.append(album)

        return albums

    async def get_album_data(self, album_id: str, db_session: Session) -> models.Album:
        """Get album data from Spotify API.

        Args:
            album_id (str): The Spotify ID of the album.
            db_session (Session): SQLAlchemy session object

        Returns:
            models.Album: The album model instance.
        """
        cache_key = f"album_{album_id}"

        if cached_data := self.cache.get(cache_key):
            return crud.album.get_by_spotify_id(
                db_session, album_id
            ) or self._create_album(db_session, cached_data)

        album_data = self.spotify.album(album_id)
        self.cache.set(cache_key, album_data)

        return crud.album.get_by_spotify_id(db_session, album_id) or self._create_album(
            db_session,
            album_data,  # type: ignore
        )

    async def get_album_tracks(
        self, album_id: str, db_session: Session
    ) -> List[models.Track]:
        cache_key = f"album_tracks_{album_id}"

        if cached_data := self.cache.get(cache_key):
            tracks = []
            for track_data in cached_data:
                try:
                    full_track_data = self.spotify.track(track_data["id"])
                    track = crud.track.get_by_spotify_id(
                        db_session, track_data["id"]
                    ) or self._create_track(
                        db_session,
                        full_track_data,  # type: ignore
                        full_track_data["artists"][0]["id"],  # type: ignore
                        album_id,  # type: ignore
                    )

                    tracks.append(track)
                except Exception as e:
                    print(
                        f"Warning: Could not process track {track_data.get('id', 'Unknown ID')}: {str(e)}"
                    )
                    continue

            return tracks

        try:
            results = self.spotify.album_tracks(album_id)
            tracks_data = results["items"]  # type: ignore

            while results["next"]:  # type: ignore
                results = self.spotify.next(results)
                tracks_data.extend(results["items"])  # type: ignore

            self.cache.set(cache_key, tracks_data)

            tracks = []
            for track_data in tracks_data:
                try:
                    full_track_data = self.spotify.track(track_data["id"])

                    track = crud.track.get_by_spotify_id(
                        db_session, track_data["id"]
                    ) or self._create_track(
                        db_session,
                        full_track_data,  # type: ignore
                        full_track_data["artists"][0]["id"],  # type: ignore
                        album_id,  # type: ignore
                    )

                    tracks.append(track)
                except Exception as e:
                    print(
                        f"Warning: Could not process track {track_data.get('id', 'Unknown ID')}: {str(e)}"
                    )
                    continue

            return tracks

        except Exception as e:
            print(f"Error fetching tracks for album {album_id}: {str(e)}")
            return []

    async def get_track_data(self, track_id: str, db_session: Session) -> models.Track:
        """Get track data from Spotify API.

        Args:
            track_id (str): The Spotify ID of the track.
            db_session (Session): SQLAlchemy session object

        Returns:
            models.Track: The track model instance.
        """
        cache_key = f"track_{track_id}"

        if cached_data := self.cache.get(cache_key):
            return crud.track.get_by_spotify_id(
                db_session, track_id
            ) or self._create_track(db_session, cached_data)  # type: ignore

        track_data = self.spotify.track(track_id)

        full_track_data = {
            **track_data,  # type: ignore
        }
        self.cache.set(cache_key, full_track_data)

        artist = await self.get_artist_data(track_data["artists"][0]["id"], db_session)  # type: ignore
        album = await self.get_album_data(track_data["album"]["id"], db_session)  # type: ignore

        track = crud.track.get_by_spotify_id(
            db_session, track_id
        ) or self._create_track(
            db_session,
            full_track_data,
            artist.id,  # type: ignore
            album.id,  # type: ignore
        )

        return track

    def _create_artist(self, db_session: Session, artist_data: Dict) -> models.Artist:
        """Create an artist model instance from Spotify data.

        Args:
            db_session (Session): SQLAlchemy session object
            artist_data (Dict): The artist data from Spotify.

        Returns:
            models.Artist: The artist model instance.
        """
        artist_create = {
            "spotify_id": artist_data["id"],
            "name": artist_data["name"],
            "popularity": artist_data["popularity"],
            "followers": artist_data["followers"]["total"],
            "image_url": artist_data["images"][0]["url"]
            if artist_data["images"]
            else None,
        }

        artist = crud.artist.create(db_session, **artist_create)

        for genre_name in artist_data["genres"]:
            genre = crud.genre.get_by_name(db_session, genre_name) or crud.genre.create(
                db_session, name=genre_name
            )
            if genre not in artist.genres:
                artist.genres.append(genre)

        db_session.commit()
        return artist

    def _create_album(self, db_session: Session, album_data: Dict) -> models.Album:
        """Create an album model instance from Spotify data.

        Args:
            db_session (Session): SQLAlchemy session object
            album_data (Dict): The album data from Spotify.

        Returns:
            models.Album: The album model instance.
        """
        album_create = {
            "spotify_id": album_data["id"],
            "name": album_data["name"],
            "album_type": album_data["album_type"],
            "total_tracks": album_data["total_tracks"],
            "release_date": album_data["release_date"],
            "image_url": album_data["images"][0]["url"]
            if album_data["images"]
            else None,
            "popularity": album_data.get("popularity", 0),
            "label": album_data.get("label"),
        }

        album = crud.album.create(db_session, **album_create)

        for artist_data in album_data["artists"]:
            artist = crud.artist.get_by_spotify_id(db_session, artist_data["id"])
            if artist and artist not in album.artists:
                album.artists.append(artist)

        db_session.commit()
        return album

    def _create_track(
        self, db_session: Session, track_data: Dict, artist_id: int, album_id: int
    ) -> models.Track:
        """Create a track model instance from Spotify data.

        Args:
            db_session (Session): SQLAlchemy session object
            track_data (Dict): The track data from Spotify.
            artist_id (int): The ID of the artist.
            album_id (int): The ID of the album.

        Returns:
            models.Track: The track model instance.
        """
        track_create = {
            "spotify_id": track_data["id"],
            "name": track_data["name"],
            "artist_id": artist_id,
            "album_id": album_id,
            "track_number": track_data["track_number"],
            "disc_number": track_data["disc_number"],
            "duration_ms": track_data["duration_ms"],
            "explicit": track_data["explicit"],
            "popularity": track_data["popularity"],
            "preview_url": track_data.get("preview_url"),
            "isrc": track_data.get("external_ids", {}).get("isrc"),
        }
        return crud.track.create(db_session, **track_create)

    def _create_audio_features(
        self, db_session: Session, track_id: int, features: Dict
    ) -> models.AudioFeatures:
        """Create an audio features model instance.

        Args:
            db_session (Session): SQLAlchemy session object
            track_id (int): The ID of the track.
            features (Dict): The audio features data.

        Returns:
            models.AudioFeatures: The audio features model instance.
        """
        feature_create = {
            "track_id": track_id,
            "acousticness": features["acousticness"],
            "danceability": features["danceability"],
            "energy": features["energy"],
            "instrumentalness": features["instrumentalness"],
            "key": features["key"],
            "liveness": features["liveness"],
            "loudness": features["loudness"],
            "mode": features["mode"],
            "speechiness": features["speechiness"],
            "tempo": features["tempo"],
            "time_signature": features["time_signature"],
            "valence": features["valence"],
        }
        return crud.audio_features.create(db_session, **feature_create)

    async def search_tracks(
        self, query: str, limit: int = 10, type: str = "track"
    ) -> List[Dict]:
        """Search for tracks on Spotify.

        Args:
            query (str): The search query.
            limit (int, optional): Limits the query response. Defaults to 10.

        Returns:
            List[Dict]: The list of track results.
        """
        cache_key = f"search_{query}_{limit}"

        if cached_results := self.cache.get(cache_key):
            return cached_results

        results = self.spotify.search(q=query, type="track", limit=limit)
        self.cache.set(cache_key, results["tracks"]["items"])  # type: ignore

        return results["tracks"]["items"]  # type: ignore

    async def get_recommendations(
        self,
        seed_tracks: Optional[List[str]] = None,
        seed_artists: Optional[List[str]] = None,
        seed_genres: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Dict]:
        """Get track recommendations from Spotify.

        Args:
            seed_tracks (Optional[List[str]], optional): The seed track IDs. Defaults to None.
            seed_artists (Optional[List[str]], optional): The seed artist IDs. Defaults to None.
            seed_genres (Optional[List[str]], optional): The seed genre names. Defaults to None.
            limit (int, optional): Limits the query response. Defaults to 20.

        Returns:
            List[Dict]: The list of recommended tracks.
        """
        cache_key = f"recs_{seed_tracks}_{seed_artists}_{seed_genres}_{limit}"

        if cached_results := self.cache.get(cache_key):
            return cached_results

        results = self.spotify.recommendations(
            seed_tracks=seed_tracks,
            seed_artists=seed_artists,
            seed_genres=seed_genres,
            limit=limit,
        )
        self.cache.set(cache_key, results["tracks"])  # type: ignore

        return results["tracks"]  # type: ignore

    async def batch_get_tracks(
        self, track_ids: List[str], db_session: Session
    ) -> List[models.Track]:
        """Get multiple tracks in batches.

        Args:
            track_ids (List[str]): The list of track IDs.
            db_session (Session): SQLAlchemy session object

        Returns:
            List[models.Track]: The list of track model instances.
        """
        tracks = []
        batch_size = 50

        for i in range(0, len(track_ids), batch_size):
            batch_ids = track_ids[i : i + batch_size]

            tracks_data = self.spotify.tracks(batch_ids)
            features_data = self.spotify.audio_features(batch_ids)

            for track_data, features in zip(tracks_data["tracks"], features_data):  # type: ignore
                if track_data is None:
                    continue

                full_data = {**track_data, "audio_features": features}  # noqa: F841
                track = await self.get_track_data(track_data["id"], db_session)
                tracks.append(track)

        return tracks

    async def update_entity(
        self, entity_type: str, spotify_id: str, db_session: Session
    ):
        """Update an entity with fresh data from Spotify.

        Args:
            entity_type (str): The type of entity to update.
            spotify_id (str): The Spotify ID of the entity.
            db_session (Session): SQLAlchemy session object

        Raises:
            ValueError: If the entity type is unknown.

        Returns:
            Multiple Types: The updated entity model instance.
        """
        if entity_type == "artist":
            data = self.spotify.artist(spotify_id)
            artist = crud.artist.get_by_spotify_id(db_session, spotify_id)
            if not artist:
                return await self.get_artist_data(spotify_id, db_session)

            artist.name = data["name"]  # type: ignore
            artist.popularity = data["popularity"]  # type: ignore
            artist.followers = data["followers"]["total"]  # type: ignore
            artist.image_url = data["images"][0]["url"] if data["images"] else None  # type: ignore

            current_genres = {genre.name for genre in artist.genres}
            new_genres = set(data["genres"])  # type: ignore

            for genre in artist.genres[:]:
                if genre.name not in new_genres:
                    artist.genres.remove(genre)

            for genre_name in new_genres - current_genres:
                genre = crud.genre.get_by_name(
                    db_session, genre_name
                ) or crud.genre.create(db_session, name=genre_name)
                artist.genres.append(genre)

        elif entity_type == "album":
            data = self.spotify.album(spotify_id)
            album = crud.album.get_by_spotify_id(db_session, spotify_id)
            if not album:
                return await self.get_album_data(spotify_id, db_session)

            album.name = data["name"]  # type: ignore
            album.album_type = data["album_type"]  # type: ignore
            album.total_tracks = data["total_tracks"]  # type: ignore
            album.release_date = data["release_date"]  # type: ignore
            album.image_url = data["images"][0]["url"] if data["images"] else None  # type: ignore
            album.popularity = data.get("popularity")  # type: ignore
            album.label = data.get("label")  # type: ignore

            current_artists = {artist.spotify_id for artist in album.artists}
            new_artists = {artist["id"] for artist in data["artists"]}  # type: ignore

            for artist in album.artists[:]:
                if artist.spotify_id not in new_artists:
                    album.artists.remove(artist)

            for artist_id in new_artists - current_artists:
                artist = await self.get_artist_data(artist_id, db_session)
                album.artists.append(artist)

        elif entity_type == "track":
            data = self.spotify.track(spotify_id)
            audio_features = self.spotify.audio_features([spotify_id])[0]  # type: ignore
            track = crud.track.get_by_spotify_id(db_session, spotify_id)
            if not track:
                return await self.get_track_data(spotify_id, db_session)

            track.name = data["name"]  # type: ignore
            track.track_number = data["track_number"]  # type: ignore
            track.disc_number = data["disc_number"]  # type: ignore
            track.duration_ms = data["duration_ms"]  # type: ignore
            track.explicit = data["explicit"]  # type: ignore
            track.popularity = data["popularity"]  # type: ignore
            track.preview_url = data.get("preview_url")  # type: ignore
            track.isrc = data.get("external_ids", {}).get("isrc")  # type: ignore

            artist = await self.get_artist_data(data["artists"][0]["id"], db_session)  # type: ignore
            album = await self.get_album_data(data["album"]["id"], db_session)  # type: ignore
            track.artist_id = artist.id
            track.album_id = album.id

            if audio_features and track.audio_features:
                for key, value in audio_features.items():
                    if hasattr(track.audio_features, key):
                        setattr(track.audio_features, key, value)
            elif audio_features:
                self._create_audio_features(db_session, track.id, audio_features)

        else:
            raise ValueError(f"Unknown entity type: {entity_type}")

        db_session.commit()
        return crud.get_by_spotify_id(db_session, entity_type, spotify_id)  # type: ignore
