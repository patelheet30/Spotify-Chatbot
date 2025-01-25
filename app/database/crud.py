from typing import List, Optional

from sqlalchemy.orm import Session

from . import models


class CRUDBase:
    """Base class for CRUD operations."""

    def __init__(self, model):
        self.model = model

    def get_by_id(self, db_session: Session, id: int):
        """Get a single record by its ID. The ID is the primary key of the table.

        Args:
            db_session (Session): SQLAlchemy session object
            id (int): ID of the record

        Returns:
            Query: SQLAlchemy query object
        """
        return db_session.query(self.model).filter(self.model.id == id).first()

    def get_by_spotify_id(self, db_session: Session, spotify_id: str):
        """Get a single record by its Spotify ID. The Spotify ID is unique for each record.

        Args:
            db_session (Session): SQLAlchemy session object
            spotify_id (str): Spotify ID of the record

        Returns:
            Query: SQLAlchemy query object
        """
        return (
            db_session.query(self.model)
            .filter(self.model.spotify_id == spotify_id)
            .first()
        )

    def get_multi(self, db_session: Session, *, skip: int = 0, limit: int = 100):
        """Get multiple records from the table.

        Args:
            db_session (Session): SQLAlchemy session object
            skip (int, optional): How many records to skip. Defaults to 0.
            limit (int, optional): How many records to collect. Defaults to 100.

        Returns:
            Query: SQLAlchemy query object
        """
        return db_session.query(self.model).offset(skip).limit(limit).all()


class CRUDArtist(CRUDBase):
    """CRUD operations for the Artist model.

    Args:
        CRUDBase (CRUDBase): Base class for CRUD operations
    """

    def __init__(self):
        super().__init__(models.Artist)

    def create(
        self,
        db_session: Session,
        *,
        spotify_id: str,
        name: str,
        popularity: Optional[int] = None,
        followers: Optional[int] = None,
        image_url: Optional[str] = None,
    ) -> models.Artist:
        """Create a new artist record in the database.

        Args:
            db_session (Session): SQLAlchemy session object
            spotify_id (str): Spotify ID of the artist
            name (str): Name of the artist
            popularity (Optional[int], optional): Popularity of the artist. Defaults to None.
            followers (Optional[int], optional): Follower count of the artist. Defaults to None.
            image_url (Optional[str], optional): Profile Image URL of the artist. Defaults to None.

        Returns:
            models.Artist: The created artist record
        """
        db_obj = models.Artist(
            spotify_id=spotify_id,
            name=name,
            popularity=popularity,
            followers=followers,
            image_url=image_url,
        )
        db_session.add(db_obj)
        db_session.commit()
        db_session.refresh(db_obj)
        return db_obj

    def update(
        self,
        db_session: Session,
        *,
        db_obj: models.Artist,
        name: Optional[str] = None,
        popularity: Optional[int] = None,
        followers: Optional[int] = None,
        image_url: Optional[str] = None,
    ) -> models.Artist:
        """Update an existing artist record in the database.

        Args:
            db_session (Session): SQLAlchemy session object
            db_obj (models.Artist): The artist record to update
            name (Optional[str], optional): The new artist name. Defaults to None.
            popularity (Optional[int], optional): The new popularity count for the artist. Defaults to None.
            followers (Optional[int], optional): The new follower count for the artist. Defaults to None.
            image_url (Optional[str], optional): The new Profile Image URL for the artist. Defaults to None.

        Returns:
            models.Artist: The updated artist record
        """
        update_data = {
            "name": name,
            "popularity": popularity,
            "followers": followers,
            "image_url": image_url,
        }
        for field, value in update_data.items():
            if value is not None:
                setattr(db_obj, field, value)

        db_session.add(db_obj)
        db_session.commit()
        db_session.refresh(db_obj)
        return db_obj

    def get_by_genre(
        self, db_session: Session, genre_name: str, *, skip: int = 0, limit: int = 100
    ) -> List[models.Artist]:
        """Get artists by genre.

        Args:
            db_session (Session): SQLAlchemy session object
            genre_name (str): Name of the genre
            skip (int, optional): How many records to skip. Defaults to 0.
            limit (int, optional): How many records to collect. Defaults to 100.

        Returns:
            List[models.Artist]: List of artists
        """
        return (
            db_session.query(self.model)
            .join(models.artist_genre)
            .join(models.Genre)
            .filter(models.Genre.name == genre_name)
            .offset(skip)
            .limit(limit)
            .all()
        )


class CRUDAlbum(CRUDBase):
    """CRUD operations for the Album model.

    Args:
        CRUDBase (CRUDBase): Base class for CRUD operations
    """

    def __init__(self):
        super().__init__(models.Album)

    def create(
        self,
        db_session: Session,
        *,
        spotify_id: str,
        name: str,
        album_type: Optional[str] = None,
        total_tracks: Optional[int] = None,
        release_date: Optional[str] = None,
        image_url: Optional[str] = None,
        popularity: Optional[int] = None,
        label: Optional[str] = None,
    ) -> models.Album:
        """Create a new album record in the database.

        Args:
            db_session (Session): SQLAlchemy session object
            spotify_id (str): Spotify ID of the album
            name (str): Name of the album
            album_type (Optional[str], optional): The type of album. Defaults to None.
            total_tracks (Optional[int], optional): Total tracks in the album. Defaults to None.
            release_date (Optional[str], optional): Release date of the album. Defaults to None.
            image_url (Optional[str], optional): Album Artwork URL of the album. Defaults to None.
            popularity (Optional[int], optional): Popularity of the album. Defaults to None.
            label (Optional[str], optional): Record/Music Label that released the album. Defaults to None.

        Returns:
            models.Album: The created album record
        """
        db_obj = models.Album(
            spotify_id=spotify_id,
            name=name,
            album_type=album_type,
            total_tracks=total_tracks,
            release_date=release_date,
            image_url=image_url,
            popularity=popularity,
            label=label,
        )
        db_session.add(db_obj)
        db_session.commit()
        db_session.refresh(db_obj)
        return db_obj

    def update(
        self,
        db_session: Session,
        *,
        db_obj: models.Album,
        name: Optional[str] = None,
        album_type: Optional[str] = None,
        total_tracks: Optional[int] = None,
        release_date: Optional[str] = None,
        image_url: Optional[str] = None,
        popularity: Optional[int] = None,
        label: Optional[str] = None,
    ) -> models.Album:
        """Update an existing album record in the database.

        Args:
            db_session (Session): SQLAlchemy session object
            db_obj (models.Album): The album record to update
            name (Optional[str], optional): The new name of the album. Defaults to None.
            album_type (Optional[str], optional): The new album type. Defaults to None.
            total_tracks (Optional[int], optional): The new track count of the album. Defaults to None.
            release_date (Optional[str], optional): The new release date of the album. Defaults to None.
            image_url (Optional[str], optional): The new album artwork URL. Defaults to None.
            popularity (Optional[int], optional): The new popularity count of the album. Defaults to None.
            label (Optional[str], optional): The new publisher of the album. Defaults to None.

        Returns:
            models.Album: The updated album record
        """
        update_data = {
            "name": name,
            "album_type": album_type,
            "total_tracks": total_tracks,
            "release_date": release_date,
            "image_url": image_url,
            "popularity": popularity,
            "label": label,
        }
        for field, value in update_data.items():
            if value is not None:
                setattr(db_obj, field, value)

        db_session.add(db_obj)
        db_session.commit()
        db_session.refresh(db_obj)
        return db_obj

    def get_by_artist(
        self, db_session: Session, artist_id: int, *, skip: int = 0, limit: int = 100
    ) -> List[models.Album]:
        """Get albums by artist.

        Args:
            db_session (Session): SQLAlchemy session object
            artist_id (int): ID of the artist
            skip (int, optional): How many records to skip. Defaults to 0.
            limit (int, optional): How many records to collect. Defaults to 100.

        Returns:
            List[models.Album]: List of albums
        """
        return (
            db_session.query(self.model)
            .join(models.album_artist)
            .filter(models.album_artist.c.artist_id == artist_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_by_year(
        self, db_session: Session, year: str, *, skip: int = 0, limit: int = 100
    ) -> List[models.Album]:
        """Get albums by year.

        Args:
            db_session (Session): SQLAlchemy session object
            year (str): Year to filter the albums
            skip (int, optional): How many records to skip. Defaults to 0.
            limit (int, optional): How many records to collect. Defaults to 100.

        Returns:
            List[models.Album]: List of albums
        """
        return (
            db_session.query(self.model)
            .filter(models.Album.release_date.like(f"{year}%"))
            .offset(skip)
            .limit(limit)
            .all()
        )


class CRUDTrack(CRUDBase):
    """CRUD operations for the Track model.

    Args:
        CRUDBase (CRUDBase): Base class for CRUD operations
    """

    def __init__(self):
        super().__init__(models.Track)

    def create(
        self,
        db_session: Session,
        *,
        spotify_id: str,
        name: str,
        artist_id: int,
        album_id: int,
        track_number: Optional[int] = None,
        disc_number: Optional[int] = None,
        duration_ms: Optional[int] = None,
        explicit: Optional[bool] = None,
        popularity: Optional[int] = None,
        preview_url: Optional[str] = None,
        isrc: Optional[str] = None,
    ) -> models.Track:
        """Create a new track record in the database.

        Args:
            db_session (Session): SQLAlchemy session object
            spotify_id (str): The Spotify ID of the track
            name (str): The name of the track
            artist_id (int): The ID of the artist
            album_id (int): The ID of the album
            track_number (Optional[int], optional): What number the track is on its album. Defaults to None.
            disc_number (Optional[int], optional): Which disc the track is part of on the album. Defaults to None.
            duration_ms (Optional[int], optional): The duration of the track in ms. Defaults to None.
            explicit (Optional[bool], optional): If the track is explicit. Defaults to None.
            popularity (Optional[int], optional): Popularity of the track. Defaults to None.
            preview_url (Optional[str], optional): The preview URL for the track. Defaults to None.
            isrc (Optional[str], optional): The International Standard Recording Code for the Album. Defaults to None.

        Returns:
            models.Track: The created track record
        """
        db_obj = models.Track(
            spotify_id=spotify_id,
            name=name,
            artist_id=artist_id,
            album_id=album_id,
            track_number=track_number,
            disc_number=disc_number,
            duration_ms=duration_ms,
            explicit=1 if explicit else 0 if explicit is not None else None,
            popularity=popularity,
            preview_url=preview_url,
            isrc=isrc,
        )
        db_session.add(db_obj)
        db_session.commit()
        db_session.refresh(db_obj)
        return db_obj

    def update(
        self,
        db_session: Session,
        *,
        db_obj: models.Track,
        name: Optional[str] = None,
        track_number: Optional[int] = None,
        disc_number: Optional[int] = None,
        duration_ms: Optional[int] = None,
        explicit: Optional[bool] = None,
        popularity: Optional[int] = None,
        preview_url: Optional[str] = None,
        isrc: Optional[str] = None,
    ) -> models.Track:
        """Update an existing track record in the database.

        Args:
            db_session (Session): SQLAlchemy session object
            db_obj (models.Track): The track record to update
            name (Optional[str], optional): The new track name. Defaults to None.
            track_number (Optional[int], optional): The new track position on the album. Defaults to None.
            disc_number (Optional[int], optional): The new disc the track is allocated to. Defaults to None.
            duration_ms (Optional[int], optional): The new duration of the track. Defaults to None.
            explicit (Optional[bool], optional): Determines if the track has changed its explicity standing. Defaults to None.
            popularity (Optional[int], optional): The new popularity of the track. Defaults to None.
            preview_url (Optional[str], optional): The new preview URL of the track. Defaults to None.
            isrc (Optional[str], optional): The new International Standard Recording Code for the track. Defaults to None.

        Returns:
            models.Track: The updated track record
        """
        update_data = {
            "name": name,
            "track_number": track_number,
            "disc_number": disc_number,
            "duration_ms": duration_ms,
            "explicit": 1 if explicit else 0 if explicit is not None else None,
            "popularity": popularity,
            "preview_url": preview_url,
            "isrc": isrc,
        }
        for field, value in update_data.items():
            if value is not None:
                setattr(db_obj, field, value)

        db_session.add(db_obj)
        db_session.commit()
        db_session.refresh(db_obj)
        return db_obj

    def get_by_artist(
        self, db_session: Session, artist_id: int, *, skip: int = 0, limit: int = 100
    ) -> List[models.Track]:
        """Get tracks by artist.

        Args:
            db_session (Session): SQLAlchemy session object
            artist_id (int): ID of the artist
            skip (int, optional): How many records to skip. Defaults to 0.
            limit (int, optional): How many records to collect. Defaults to 100.

        Returns:
            List[models.Track]: List of tracks
        """
        return (
            db_session.query(self.model)
            .filter(self.model.artist_id == artist_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_by_album(
        self, db_session: Session, album_id: int, *, skip: int = 0, limit: int = 100
    ) -> List[models.Track]:
        """Get tracks by album.

        Args:
            db_session (Session): SQLAlchemy session object
            album_id (int): ID of the album
            skip (int, optional): How many records to skip. Defaults to 0.
            limit (int, optional): How many records to collect. Defaults to 100.

        Returns:
            List[models.Track]: List of tracks
        """
        return (
            db_session.query(self.model)
            .filter(self.model.album_id == album_id)
            .order_by(self.model.disc_number, self.model.track_number)
            .offset(skip)
            .limit(limit)
            .all()
        )


class CRUDAudioFeatures(CRUDBase):
    """CRUD operations for the AudioFeatures model.

    Args:
        CRUDBase (CRUDBase): Base class for CRUD operations
    """

    def __init__(self):
        super().__init__(models.AudioFeatures)

    def create(
        self,
        db_session: Session,
        *,
        track_id: int,
        acousticness: Optional[float] = None,
        danceability: Optional[float] = None,
        energy: Optional[float] = None,
        instrumentalness: Optional[float] = None,
        key: Optional[int] = None,
        liveness: Optional[float] = None,
        loudness: Optional[float] = None,
        mode: Optional[int] = None,
        speechiness: Optional[float] = None,
        tempo: Optional[float] = None,
        time_signature: Optional[int] = None,
        valence: Optional[float] = None,
    ) -> models.AudioFeatures:
        """Create a new audio features record in the database.

        Args:
            db_session (Session): SQLAlchemy session object
            track_id (int): ID of the track
            acousticness (Optional[float], optional): The acousticness of the song. Defaults to None.
            danceability (Optional[float], optional): The danceability of the song. Defaults to None.
            energy (Optional[float], optional): The energy of the song. Defaults to None.
            instrumentalness (Optional[float], optional): The instrumentalness of the song. Defaults to None.
            key (Optional[int], optional): The key of the song. Defaults to None.
            liveness (Optional[float], optional): The liveness of the song. Defaults to None.
            loudness (Optional[float], optional): The loudness of the song. Defaults to None.
            mode (Optional[int], optional): The mode of the song. Defaults to None.
            speechiness (Optional[float], optional): The speechiness of the song. Defaults to None.
            tempo (Optional[float], optional): The tempo of the song. Defaults to None.
            time_signature (Optional[int], optional): The time signature of the song. Defaults to None.
            valence (Optional[float], optional): The valence of the song. Defaults to None.

        Returns:
            models.AudioFeatures: The created audio features record
        """
        db_obj = models.AudioFeatures(
            track_id=track_id,
            acousticness=acousticness,
            danceability=danceability,
            energy=energy,
            instrumentalness=instrumentalness,
            key=key,
            liveness=liveness,
            loudness=loudness,
            mode=mode,
            speechiness=speechiness,
            tempo=tempo,
            time_signature=time_signature,
            valence=valence,
        )
        db_session.add(db_obj)
        db_session.commit()
        db_session.refresh(db_obj)
        return db_obj

    def update(
        self,
        db_session: Session,
        *,
        db_obj: models.AudioFeatures,
        acousticness: Optional[float] = None,
        danceability: Optional[float] = None,
        energy: Optional[float] = None,
        instrumentalness: Optional[float] = None,
        key: Optional[int] = None,
        liveness: Optional[float] = None,
        loudness: Optional[float] = None,
        mode: Optional[int] = None,
        speechiness: Optional[float] = None,
        tempo: Optional[float] = None,
        time_signature: Optional[int] = None,
        valence: Optional[float] = None,
    ) -> models.AudioFeatures:
        """Update an existing audio features record in the database.

        Args:
            db_session (Session): SQLAlchemy session object
            db_obj (models.AudioFeatures): The audio features record to update
            acousticness (Optional[float], optional): The new acousticness of the song. Defaults to None.
            danceability (Optional[float], optional): The new danceability of the song. Defaults to None.
            energy (Optional[float], optional): The new energy of the song. Defaults to None.
            instrumentalness (Optional[float], optional): The new instrumentalness of the song. Defaults to None.
            key (Optional[int], optional): The new key of the song. Defaults to None.
            liveness (Optional[float], optional): The new liveness of the song. Defaults to None.
            loudness (Optional[float], optional): The new loudness of the song. Defaults to None.
            mode (Optional[int], optional): The new mode of the song. Defaults to None.
            speechiness (Optional[float], optional): The new speechiness of the song. Defaults to None.
            tempo (Optional[float], optional): The new tempo of the song. Defaults to None.
            time_signature (Optional[int], optional): The new time signature of the song. Defaults to None.
            valence (Optional[float], optional): The new valence of the song. Defaults to None.

        Returns:
            models.AudioFeatures: The updated audio features record
        """
        update_data = {
            "acousticness": acousticness,
            "danceability": danceability,
            "energy": energy,
            "instrumentalness": instrumentalness,
            "key": key,
            "liveness": liveness,
            "loudness": loudness,
            "mode": mode,
            "speechiness": speechiness,
            "tempo": tempo,
            "time_signature": time_signature,
            "valence": valence,
        }
        for field, value in update_data.items():
            if value is not None:
                setattr(db_obj, field, value)

        db_session.add(db_obj)
        db_session.commit()
        db_session.refresh(db_obj)
        return db_obj


class CRUDGenre(CRUDBase):
    """CRUD operations for the Genre model

    Args:
        CRUDBase (CRUDBase): Base class for CRUD operations
    """

    def __init__(self):
        super().__init__(models.Genre)

    def create(self, db_session: Session, *, name: str) -> models.Genre:
        """Create a new genre record in the database.

        Args:
            db_session (Session): SQLAlchemy session object
            name (str): Name of the genre

        Returns:
            models.Genre: The created genre record
        """
        db_obj = models.Genre(name=name)
        db_session.add(db_obj)
        db_session.commit()
        db_session.refresh(db_obj)
        return db_obj

    def update(
        self, db_session: Session, *, db_obj: models.Genre, name: Optional[str] = None
    ) -> models.Genre:
        """Update an existing genre record in the database.

        Args:
            db_session (Session): SQLAlchemy session object
            db_obj (models.Genre): The genre record to update
            name (Optional[str], optional): The new name of the genre. Defaults to None.

        Returns:
            models.Genre: The updated genre record
        """
        update_data = {"name": name}
        for field, value in update_data.items():
            if value is not None:
                setattr(db_obj, field, value)

        db_session.add(db_obj)
        db_session.commit()
        db_session.refresh(db_obj)
        return db_obj

    def get_by_name(self, db_session: Session, name: str) -> Optional[models.Genre]:
        """Get a genre record by its name.

        Args:
            db_session (Session): SQLAlchemy session object
            name (str): Name of the genre

        Returns:
            Optional[models.Genre]: The genre record if found, else None
        """
        return db_session.query(self.model).filter(self.model.name == name).first()


artist = CRUDArtist()
album = CRUDAlbum()
track = CRUDTrack()
audio_features = CRUDAudioFeatures()
genre = CRUDGenre()
