from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


artist_genre = Table(
    "artist_genre",
    Base.metadata,
    Column("artist_id", Integer, ForeignKey("artists.id")),
    Column("genre_id", Integer, ForeignKey("genres.id")),
)

album_artist = Table(
    "album_artist",
    Base.metadata,
    Column("album_id", Integer, ForeignKey("albums.id")),
    Column("artist_id", Integer, ForeignKey("artists.id")),
)


class Artist(Base):
    __tablename__ = "artists"

    id = Column(Integer, primary_key=True)
    spotify_id = Column(String, unique=True, index=True)
    name = Column(String, nullable=False)
    popularity = Column(Integer)
    followers = Column(Integer)
    image_url = Column(String)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    tracks = relationship("Track", back_populates="artist")
    albums = relationship("Album", secondary=album_artist, back_populates="artists")
    genres = relationship("Genre", secondary=artist_genre, back_populates="artists")


class Album(Base):
    __tablename__ = "albums"

    id = Column(Integer, primary_key=True)
    spotify_id = Column(String, unique=True, index=True)
    name = Column(String, nullable=False)
    album_type = Column(String)
    total_tracks = Column(Integer)
    release_date = Column(String)
    image_url = Column(String)
    popularity = Column(Integer)
    label = Column(String)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    artists = relationship("Artist", secondary=album_artist, back_populates="albums")
    tracks = relationship("Track", back_populates="album")


class Track(Base):
    __tablename__ = "tracks"

    id = Column(Integer, primary_key=True)
    spotify_id = Column(String, unique=True, index=True)
    name = Column(String, nullable=False)
    artist_id = Column(Integer, ForeignKey("artists.id"))
    album_id = Column(Integer, ForeignKey("albums.id"))
    track_number = Column(Integer)
    disc_number = Column(Integer)
    duration_ms = Column(Integer)
    explicit = Column(Integer)
    popularity = Column(Integer)
    preview_url = Column(String)
    isrc = Column(String)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    artist = relationship("Artist", back_populates="tracks")
    album = relationship("Album", back_populates="tracks")
    audio_features = relationship(
        "AudioFeatures", back_populates="track", uselist=False
    )


class AudioFeatures(Base):
    __tablename__ = "audio_features"

    id = Column(Integer, primary_key=True)
    track_id = Column(Integer, ForeignKey("tracks.id"), unique=True)
    acousticness = Column(Float)
    danceability = Column(Float)
    energy = Column(Float)
    instrumentalness = Column(Float)
    key = Column(Integer)
    liveness = Column(Float)
    loudness = Column(Float)
    mode = Column(Integer)
    speechiness = Column(Float)
    tempo = Column(Float)
    time_signature = Column(Integer)
    valence = Column(Float)
    created_at = Column(DateTime, server_default=func.now())

    track = relationship("Track", back_populates="audio_features")


class Genre(Base):
    __tablename__ = "genres"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    artists = relationship("Artist", secondary=artist_genre, back_populates="genres")
