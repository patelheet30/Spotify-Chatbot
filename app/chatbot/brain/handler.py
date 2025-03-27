import csv
import logging
import os
import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import aiml
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.api.spotify import SpotifyAPI
from app.chatbot.brain.logic import KnowledgeBase
from app.chatbot.vision.classifier import MusicAlbumClassifier
from app.chatbot.vision.utils import get_image_path
from app.chatbot.voice import (
    get_speech_recogniser,
    get_temp_audio_file,
    get_text_to_speech,
    play_audio_file,
)
from app.database import crud

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    NORMAL = 0
    AWAITING_IMAGE = 1
    AWAITING_VOICE_INPUT = 2
    AWAITING_ENSEMBLE_CHOICE = 3


class MusicInfoChatbot:
    """
    todo: Work on credits for the info and response handler and also listeners
    todo: Remove templates for tempo, genre etc.
    """

    def __init__(self, db_session, spotify_api: SpotifyAPI):
        self.db_session = db_session
        self.spotify_api = spotify_api

        self.kernel = aiml.Kernel()
        aiml_path = os.path.join(
            os.path.dirname(__file__), "..", "knowledge", "patterns.xml"
        )
        self.kernel.learn(aiml_path)

        self.qa_pairs = self._load_qa_pairs()

        self.kb = KnowledgeBase()
        logger.info(
            f"Loaded {len(self.kb.kb_expressions)} statements from knowledge base"
        )

        self.classifier = MusicAlbumClassifier()
        logger.info(
            f"Loaded {len(self.classifier.models)} models for music album classification"
        )

        self.speech_recogniser = get_speech_recogniser()
        self.text_to_speech = get_text_to_speech()
        logger.info(f"Speech Recogniser: {self.speech_recogniser.is_available()}")
        logger.info(f"Text to Speech: {self.text_to_speech.is_available()}")
        self.voice_enabled = False
        self.voice_available = (
            self.speech_recogniser.is_available() and self.text_to_speech.is_available()
        )

        if self.voice_available:
            logger.info("Voice recognition and text-to-speech are available.")
        else:
            logger.warning("Voice recognition or text-to-speech is not available.")

        self.state = ConversationState.NORMAL
        self.context = {}

        nltk.download("stopwords", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        nltk.download("wordnet", quiet=True)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

        self.vectorizer = TfidfVectorizer()
        self._prepare_similarity_matching()

    def _load_qa_pairs(self) -> Dict[str, str]:
        qa_dict = {}
        qa_path = os.path.join(
            os.path.dirname(__file__), "..", "knowledge", "qa_pairs.csv"
        )

        try:
            with open(qa_path, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    qa_dict[row["question"]] = row["answer"]
                return qa_dict
        except FileNotFoundError:
            print(f"File not found: {qa_path}")
            return {}

    def _prepare_similarity_matching(self):
        if not self.qa_pairs:
            self.questions = []
            self.tfidf_matrix = None
            return

        questions = list(self.qa_pairs.keys())
        self.tfidf_matrix = self.vectorizer.fit_transform(questions)  # type: ignore
        self.questions = questions

    def _preprocess_text(self, text: str) -> str:
        tokens = word_tokenize(text.lower())
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 1
        ]
        return " ".join(tokens)

    def _get_similar_question(
        self, user_input: str, threshold: float = 0.3
    ) -> Optional[str]:
        if self.tfidf_matrix is None or self.tfidf_matrix.shape[0] == 0:
            return None

        cleaned_input = self._clean_input(user_input.lower())

        logger.info(f"Finding similar question for: {cleaned_input}")

        if cleaned_input.startswith("what is") or cleaned_input.startswith("what are"):
            subject = (
                cleaned_input.replace("what is", "").replace("what are", "").strip()
            )

            for question in self.questions:
                q_lower = question.lower()
                if subject in q_lower and (
                    q_lower.startswith("what is")
                    or q_lower.startswith("what are")
                    or "define" in q_lower
                ):
                    logger.info(
                        f"Direct match found for subject: {subject} in question: {question}"
                    )
                    return question

        processed_input = self._preprocess_text(cleaned_input)
        input_vector = self.vectorizer.transform([processed_input])
        similarities = cosine_similarity(input_vector, self.tfidf_matrix)[0]
        max_similarity_idx = np.argmax(similarities)
        max_similarity = similarities[max_similarity_idx]

        logger.info(
            f"Best match: {self.questions[max_similarity_idx]} with score {max_similarity}"
        )

        if max_similarity >= threshold:
            return self.questions[max_similarity_idx]

        return None

    def _clean_input(self, text: str) -> str:
        cleaned = re.sub(r"[?.!,;:]", "", text)
        cleaned = " ".join(cleaned.split())
        return cleaned.strip()

    async def classify_album_era(
        self, img_path: str, use_ensemble: bool = False
    ) -> Dict[str, Any]:
        try:
            img_path = get_image_path(img_path)  # type: ignore
            if not img_path:
                return {
                    "query_type": "album_era_classification",
                    "era": "Invalid image path",
                }

            available_models = list(self.classifier.models.keys())

            logger.info(f"Available models: {', '.join(available_models)}")
            logger.info(f"Ensemble classification: {str(use_ensemble)}")

            result = self.classifier.classify_image(img_path, ensemble=use_ensemble)

            confidence = f"{result['confidence'] * 100:.1f}%"

            if "model_used" in result:
                if result["model_used"] == "ensemble":
                    model_info = f"Ensemble of {', '.join(available_models)}"
                else:
                    model_info = f"Model used: {result['model_used']}"
            else:
                model_info = (
                    "available model"
                    if len(available_models) == 1
                    else "ensemble model"
                )

            context = {
                "query_type": "album_era_classification",
                "classification": result,
                "era": result["era"],
                "confidence": confidence,
                "model": model_info,
                "available_models": available_models,
            }
            return context
        except Exception as e:
            logger.error(f"Error classifying album era: {e}")
            return {
                "query_type": "album_era_classification",
                "era": "Error classifying album era",
                "error": str(e),
            }

    def handle_ensemble_choice(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        cleaned_input = user_input.upper().strip()
        use_ensemble = "YES" in cleaned_input or "Y" in cleaned_input

        available_models = list(self.classifier.models.keys())
        can_use_ensemble = len(available_models) > 1
        if use_ensemble and not can_use_ensemble:
            model_name = available_models[0] if available_models else "Unknown"
            message = f"I can only use the {model_name} model for classification."
            use_ensemble = False
        else:
            message = "Please provide the image (URL/PATH) of the album cover."

        self.context["use_ensemble"] = use_ensemble  # type: ignore
        self.context["available_models"] = available_models  # type: ignore
        self.state = ConversationState.AWAITING_IMAGE

        return message, self.context

    async def listen_voice(self) -> Tuple[str, Dict[str, Any]]:
        if not self.voice_available:
            return "Voice Recognition or Text-to-Speech is not available.", {}

        self.state = ConversationState.AWAITING_VOICE_INPUT

        text = self.speech_recogniser.listen()
        self.state = ConversationState.NORMAL

        if text:
            response, context = await self._process_input(text)

            if self.voice_enabled:
                audio_path = get_temp_audio_file()
                success = self.text_to_speech.save_to_file(response, audio_path)
                if success:
                    play_audio_file(audio_path)
                    context[audio_path] = audio_path
                else:
                    logger.error("Failed to save audio file.")

            return f"I heard: {text}\nMy Response: {response}", context
        else:
            return "Sorry, I couldn't hear you. Please try again.", {}

    def speak_response(self, text: str) -> Tuple[str, Dict[str, Any]]:
        if not self.voice_available:
            return "Sorry, my speech system is not enabled", {}

        try:
            audio_path = get_temp_audio_file()
            success = self.text_to_speech.save_to_file(text, audio_path)

            if success:
                self.text_to_speech.speak(text)
                return f"Speaking: {text}", {"audio_response": audio_path}
            else:
                return "Failed to generate speech.", {}
        except Exception as e:
            logger.error(f"Error in text-to-speech: {e}")
            return f"Error in speech synthesis: {str(e)}", {}

    def toggle_voice(self, enable: bool) -> Tuple[str, Dict[str, Any]]:
        if enable and not self.voice_available:
            return "Voice recognition or text-to-speech is not available.", {
                "voice_enabled": False
            }

        self.voice_enabled = enable if self.voice_available else False

        if self.voice_enabled:
            message = "Voice recognition and text-to-speech are enabled."
            audio_path = get_temp_audio_file()
            self.text_to_speech.save_to_file(message, audio_path)
            self.text_to_speech.speak(message)
            return message, {"voice_enabled": True, "audio_response": audio_path}
        else:
            return "Voice recognition and text-to-speech are disabled.", {
                "voice_enabled": False
            }

    async def _fetch_music_info(self, response: str, user_input: str) -> Dict[str, Any]:
        context = {}

        logger.info(f"Fetching music info for: {response}, user input: {user_input}")

        if "Hi! I'm your music chatbot" in response:
            context["query_type"] = "greeting"

        if "information about" in response.lower():
            artist_name = self._clean_input(
                user_input.replace("WHO IS", "").replace("TELL ME ABOUT", "").strip()
            )

            logger.info(f"Fetching artist info for: {artist_name}")

            artists = crud.artist.get_multi(self.db_session)
            artist_match = next(
                (a for a in artists if a.name.lower() == artist_name.lower()), None
            )

            if artist_match:
                context["artist_info"] = artist_match
            else:
                search_results = await self.spotify_api.search_tracks(
                    artist_name, limit=1, type="artist"
                )
                if (
                    search_results
                    and search_results[0].get("artists")
                    and search_results[0]["artists"][0].get("id")
                ):
                    artist_id = search_results[0]["artists"][0]["id"]
                    artist_data = await self.spotify_api.get_artist_data(
                        artist_id, self.db_session
                    )
                    if artist_data:
                        context["artist_info"] = artist_data

        elif "album" in response.lower():
            if "HOW MANY TRACKS" in user_input:
                album_name = (
                    response.split("Let me check the track count for")[1]
                    .split("album")[0]
                    .strip()
                )

                logger.info(f"Album name: {album_name}")

                albums = crud.album.get_multi(self.db_session, limit=2000)
                album_match = next(
                    (a for a in albums if a.name.lower() == album_name.lower()), None
                )

                if album_match:
                    context["album_info"] = album_match
                    context["query_type"] = "album_length"
                else:
                    search_results = await self.spotify_api.search_tracks(
                        album_name, limit=1, type="album"
                    )
                    if (
                        search_results
                        and search_results[0].get("album")
                        and search_results[0]["album"].get("id")
                    ):
                        album_id = search_results[0]["album"]["id"]
                        album_data = await self.spotify_api.get_album_data(
                            album_id, self.db_session
                        )
                        if album_data:
                            context["album_info"] = album_data
                            context["query_type"] = "album_length"

            elif "RELEASE IN" in user_input:
                artist, year = user_input.split("RELEASE IN")
                artist = self._clean_input(
                    artist.replace("WHICH ALBUM DID", "").strip()
                )
                year = self._clean_input(year.strip())

                artists = crud.artist.get_multi(self.db_session)
                artist_match = next(
                    (a for a in artists if a.name.lower() == artist.lower()), None
                )

                if artist_match:
                    artist_albums = artist_match.albums
                    year_albums = [
                        album
                        for album in artist_albums
                        if album.release_date and album.release_date.startswith(year)
                    ]

                    context["album_year_info"] = {
                        "artist": artist_match,
                        "year": year,
                        "albums": year_albums,
                    }
                else:
                    search_results = await self.spotify_api.search_tracks(
                        artist, limit=1
                    )
                    if (
                        search_results
                        and search_results[0].get("artists")
                        and search_results[0]["artists"][0].get("id")
                    ):
                        artist_id = search_results[0]["artists"][0]["id"]
                        artist_data = await self.spotify_api.get_artist_data(
                            artist_id, self.db_session
                        )
                        if artist_data:
                            context["album_year_info"] = {
                                "artist": artist_data,
                                "year": year,
                            }

            elif "WHEN WAS THE ALBUM" in user_input:
                # Extract the album name from the user input "WHEN WAS THE ALBUM * RELEASED"
                album_name = self._clean_input(
                    user_input.replace("WHEN WAS THE ALBUM", "").replace("RELEASED", "")
                )

                logger.info(f"Fetching album info for: {album_name}")

                albums = crud.album.get_multi(self.db_session, limit=2000)
                album_match = next(
                    (a for a in albums if a.name.lower() == album_name.lower()), None
                )

                if album_match:
                    context["album_info"] = album_match
                    context["query_type"] = "release_date"
                else:
                    search_results = await self.spotify_api.search_tracks(
                        album_name, limit=1, type="album"
                    )
                    if (
                        search_results
                        and search_results[0].get("album")
                        and search_results[0]["album"].get("id")
                    ):
                        album_id = search_results[0]["album"]["id"]
                        album_data = await self.spotify_api.get_album_data(
                            album_id, self.db_session
                        )
                        if album_data:
                            context["album_info"] = album_data
                            context["query_type"] = "release_date"

            else:
                album_name = self._clean_input(
                    user_input.split("ALBUM")[1].strip()
                    if "ALBUM" in user_input
                    else user_input.strip()
                )

                albums = crud.album.get_multi(self.db_session, limit=2000)
                album_match = next(
                    (a for a in albums if a.name.lower() == album_name.lower()), None
                )

                if album_match:
                    context["album_info"] = album_match
                else:
                    search_results = await self.spotify_api.search_tracks(
                        album_name, limit=1, type="album"
                    )
                    if (
                        search_results
                        and search_results[0].get("album")
                        and search_results[0]["album"].get("id")
                    ):
                        album_id = search_results[0]["album"]["id"]
                        album_data = await self.spotify_api.get_album_data(
                            album_id, self.db_session
                        )
                        if album_data:
                            context["album_info"] = album_data

        elif "SONGS ARE ON" in user_input or "TRACKS ON" in user_input:
            album_name = self._clean_input(
                re.sub(
                    r".*?(SONGS ARE ON|TRACKS ON|TRACKS ARE ON)",
                    "",
                    user_input,
                    flags=re.IGNORECASE,
                ).strip()
            )

            logger.info(f"Fetching tracks for album: {album_name}")

            albums = crud.album.get_multi(self.db_session, limit=2000)
            album_match = next(
                (a for a in albums if a.name.lower() == album_name.lower()), None
            )

            if album_match:
                album_tracks = crud.album.get_album_tracks(
                    self.db_session, album_match.spotify_id
                )
                logger.info(f"Found {len(album_tracks)} tracks for album: {album_name}")
                context["album_info"] = album_match
                context["album_tracks"] = album_tracks
                context["query_type"] = "track_list"
            else:
                search_results = await self.spotify_api.search_tracks(
                    album_name, limit=1, type="album"
                )
                if (
                    search_results
                    and search_results[0].get("album")
                    and search_results[0]["album"].get("id")
                ):
                    album_id = search_results[0]["album"]["id"]
                    album_data = await self.spotify_api.get_album_data(
                        album_id, self.db_session
                    )
                    if album_data:
                        context["album_info"] = album_data
                        context["query_type"] = "track_list"

        elif "HOW LONG IS" in user_input:
            track_name = self._clean_input(
                user_input.replace("HOW LONG IS", "").strip()
            )

            tracks = crud.track.get_multi(self.db_session, limit=10000)
            track_match = next(
                (t for t in tracks if t.name.lower() == track_name.lower()), None
            )

            if track_match:
                context["track_info"] = track_match
                context["query_type"] = "duration"
            else:
                search_results = await self.spotify_api.search_tracks(
                    track_name, limit=1
                )
                if search_results and search_results[0].get("id"):
                    track_id = search_results[0]["id"]
                    track_data = await self.spotify_api.get_track_data(
                        track_id, self.db_session
                    )
                    if track_data:
                        context["track_info"] = track_data
                        context["query_type"] = "duration"

        return context

    def _format_response(self, template: str, context: Dict[str, Any]) -> str:
        if context.get("query_type") == "greeting":
            return "Hello! I'm your music chatbot. You can ask me questions about artists, albums, tracks, genres, and more."

        if "artist_info" in context:
            artist = context["artist_info"]

            artist_albums = artist.albums
            artist_tracks = []
            for album in artist_albums:
                album_tracks = crud.album.get_album_tracks(
                    self.db_session, album.spotify_id
                )
                artist_tracks.extend(album_tracks)

            return f"Here's what I found about {artist.name}:\nThey have {artist.followers:,} followers on Spotify.\nTheir Spotify Popularity score is {artist.popularity}.\nThey have released {len(artist.albums)} albums and {len(artist_tracks)} tracks."

        elif "track_info" in context:
            track = context["track_info"]

            if context.get("query_type") == "duration":
                return f"The track {track.name} is {track.duration_ms // 60000} minutes and {track.duration_ms % 60000 // 1000} seconds long."

            return f"The track {track.name} is by {track.artist.name} and is part of the album {track.album.name}."

        elif "album_info" in context:
            album = context["album_info"]

            if context.get("query_type") == "album_length":
                return f"The album {album.name} has {album.total_tracks} tracks."

            if context.get("query_type") == "track_list":
                tracks = context.get("album_tracks", [])
                track_list = "\n".join(
                    [f"{track.track_number}. {track.name}" for track in tracks]
                )
                return f"The album {album.name} has {album.total_tracks} tracks:\n{track_list}"

            if context.get("query_type") == "release_date":
                release_date = datetime.strptime(
                    album.release_date, "%Y-%m-%d"
                ).strftime("%d %B %Y")
                if album.release_date:
                    return f"The album {album.name} was released on {release_date}."
                else:
                    return (
                        f"I couldn't find the release date for the album {album.name}."
                    )

            return f"The album {album.name} by {album.artists[0].name} was released on {album.release_date}."

        elif "album_year_info" in context:
            info = context["album_year_info"]
            artist = info["artist"]
            year = info["year"]

            if "albums" in info:
                year_albums = info["albums"]
            else:
                year_albums = [
                    album
                    for album in artist.albums
                    if album.release_date and album.release_date.startswith(year)
                ]

            if year_albums:
                album_list = [album for album in year_albums]
                album_name_list = "\n".join(
                    [
                        f"{i + 1}. {album.name} ({album.album_type.capitalize()})"
                        for i, album in enumerate(album_list)
                    ]
                )
                return f"{artist.name} released these albums in {year}:\n{album_name_list}."
            return f"I couldn't find any albums by {artist.name} released in {year}."

        return template

    async def _process_input(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        cleaned_input = user_input.upper().strip()

        if self.state == ConversationState.AWAITING_ENSEMBLE_CHOICE:
            return self.handle_ensemble_choice(user_input)

        elif self.state == ConversationState.AWAITING_IMAGE:
            logger.info(f"Classifying album era for image: {user_input}")
            use_ensemble = self.context.get("use_ensemble", False)
            self.state = ConversationState.NORMAL

            context = await self.classify_album_era(user_input, use_ensemble)  # type: ignore
            if "error" in context:
                return context["error"], {}

            era = context["era"]
            confidence = context["confidence"]
            model = context["model"]
            response = f"The album cover is classified as {era} with a confidence of {confidence} using the {model}."
            if era == "Classical Era":
                response += " Classical Era albums are typically from the 1980s to the 2000s, where physical media such as CDs were popular."
            elif era == "Digital Era":
                response += " Digital Era albums are typically from the 2010s to present, where digital streaming and downloads are common."

            return response, context

        elif (
            self._clean_input(cleaned_input) == "WHICH ERA IS THIS ALBUM FROM"
            or "ALBUM ERA" in cleaned_input
            or "CLASSIFY THIS ALBUM" in cleaned_input
            or "CLASSIFY ALBUM" in cleaned_input
            or "WHAT ERA IS THIS ALBUM" in cleaned_input
        ):
            self.state = ConversationState.AWAITING_ENSEMBLE_CHOICE
            self.context = {"query_type": "album_era_classification"}
            return (
                "Do you want to use ensemble classification? (YES/NO)",
                self.context,
            )

        if user_input.lower().startswith(
            "i know that"
        ) or user_input.lower().startswith("check that"):
            logger.info(f"Checking FOL: {user_input}")
            response = self.kb.handle_logic_command(user_input)
            return response, {"query_type": "logic"}

        if cleaned_input == "SAVE KNOWLEDGE BASE":
            success = self.kb.save_kb()
            if success:
                return "I've saved the knowledge base.", {}
            else:
                return "Error saving knowledge base.", {}

        if cleaned_input == "DEBUG KNOWLEDGE BASE":
            kb_info = self.kb.handle_logic_command(user_input)
            return f"Knowledge Base Info: {kb_info}", {}

        aiml_response = self.kernel.respond(cleaned_input)

        logger.info(f"AIML response: {aiml_response}")

        if aiml_response and aiml_response.startswith("#30$"):
            user_input = user_input.split("#30$")[-1].strip()
            similar_question = self._get_similar_question(user_input)
            if similar_question:
                logger.info(f"Found similar question: {similar_question}")
                return self.qa_pairs[similar_question], {}

        if (
            aiml_response
            and aiml_response != "I'll try to find relevant music information about:"
        ):
            context = await self._fetch_music_info(aiml_response, cleaned_input)
            if context:
                response = self._format_response(aiml_response, context)
                return response, context

        return "I'm sorry, I don't have an answer to that question.", {}

    async def get_response(self, user_input: str) -> Tuple[str, Dict[str, Any]]:
        cleaned_input = user_input.upper().strip()

        if (
            cleaned_input == "LISTEN TO MY VOICE"
            or "LISTEN TO ME" in cleaned_input
            or "VOICE INPUT" in cleaned_input
        ):
            return await self.listen_voice()

        if (
            cleaned_input.startswith("SAY ")
            or cleaned_input.startswith("SPEAK ")
            or cleaned_input == "SPEAK THIS"
        ):
            text_to_speech = user_input
            if cleaned_input.startswith("SAY "):
                text_to_speech = user_input[4:].strip()
            elif cleaned_input.startswith("SPEAK "):
                text_to_speech = user_input[6:].strip()
            elif cleaned_input == "SPEAK THIS":
                text_to_speech = "Hello, I am your music chatbot assistant. How can I help you today?"

            return self.speak_response(text_to_speech)

        if cleaned_input.startswith("ENABLE VOICE") or cleaned_input == "TURN ON VOICE":
            return self.toggle_voice(True)

        if (
            cleaned_input.startswith("DISABLE VOICE")
            or cleaned_input == "TURN OFF VOICE"
        ):
            return self.toggle_voice(False)

        if cleaned_input == "IS VOICE AVAILABLE":
            if self.voice_available:
                return "Yes, voice capabilities are available on this system.", {
                    "voice_available": True
                }
            else:
                return "No, voice capabilities are not available on this system.", {
                    "voice_available": False
                }

        response, context = await self._process_input(user_input)

        if self.voice_enabled and "audio_response" not in context:
            audio_path = get_temp_audio_file()
            success = self.text_to_speech.save_to_file(response, audio_path)
            if success:
                play_audio_file(audio_path)
                context["audio_response"] = audio_path

        return response, context
