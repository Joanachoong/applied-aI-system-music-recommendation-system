import csv
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    preferred_valence: float
    preferred_acousticness: float


def _gaussian(song_val: float, pref_val: float, sigma: float = 0.20) -> float:
    """Gaussian similarity: 1.0 when values match, exponentially lower as they diverge."""
    return math.exp(-((song_val - pref_val) ** 2) / (2 * sigma ** 2))


def _score_song_dict(song: Dict, user_prefs: Dict) -> float:
    """Score a song dict against a user_prefs dict. Returns float in [0, 1]."""
    valence_score      = _gaussian(song["valence"],      user_prefs["preferred_valence"])
    mood_score         = 1.0 if song["mood"] == user_prefs["favorite_mood"] else 0.0
    energy_score       = _gaussian(song["energy"],       user_prefs["target_energy"])
    genre_score        = 1.0 if song["genre"] == user_prefs["favorite_genre"] else 0.0
    acousticness_score = _gaussian(song["acousticness"], user_prefs["preferred_acousticness"])

    return (0.30 * valence_score
            + 0.25 * mood_score
            + 0.20 * energy_score
            + 0.15 * genre_score
            + 0.10 * acousticness_score)


def _score_song_obj(song: Song, user: "UserProfile") -> float:
    """Score a Song dataclass against a UserProfile dataclass. Returns float in [0, 1]."""
    valence_score      = _gaussian(song.valence,      user.preferred_valence)
    mood_score         = 1.0 if song.mood == user.favorite_mood else 0.0
    energy_score       = _gaussian(song.energy,       user.target_energy)
    genre_score        = 1.0 if song.genre == user.favorite_genre else 0.0
    acousticness_score = _gaussian(song.acousticness, user.preferred_acousticness)

    return (0.30 * valence_score
            + 0.25 * mood_score
            + 0.20 * energy_score
            + 0.15 * genre_score
            + 0.10 * acousticness_score)


def _explain_song_dict(song: Dict, user_prefs: Dict) -> str:
    """Return a one-line explanation naming the top-contributing feature."""
    contributions = {
        "valence":      0.30 * _gaussian(song["valence"],      user_prefs["preferred_valence"]),
        "mood":         0.25 * (1.0 if song["mood"] == user_prefs["favorite_mood"] else 0.0),
        "energy":       0.20 * _gaussian(song["energy"],       user_prefs["target_energy"]),
        "genre":        0.15 * (1.0 if song["genre"] == user_prefs["favorite_genre"] else 0.0),
        "acousticness": 0.10 * _gaussian(song["acousticness"], user_prefs["preferred_acousticness"]),
    }
    top_feature = max(contributions, key=contributions.get)

    if top_feature == "mood":
        return f"Matched your favorite mood: {song['mood']}"
    if top_feature == "genre":
        return f"Matched your favorite genre: {song['genre']}"

    pref_map = {
        "valence":      user_prefs["preferred_valence"],
        "energy":       user_prefs["target_energy"],
        "acousticness": user_prefs["preferred_acousticness"],
    }
    return (f"Matched your {top_feature} preference "
            f"({song[top_feature]:.2f} vs your {pref_map[top_feature]:.2f})")


class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        scored = sorted(self.songs, key=lambda s: _score_song_obj(s, user), reverse=True)

        result: List[Song] = []
        genre_counts: Dict[str, int] = {}
        last_mood: str = ""

        for song in scored:
            if len(result) >= k:
                break
            if genre_counts.get(song.genre, 0) >= 2:
                continue
            # If score is within 0.05 of the last added song, prefer a different mood
            if result:
                last_score = _score_song_obj(result[-1], user)
                this_score = _score_song_obj(song, user)
                if abs(this_score - last_score) <= 0.05 and song.mood == last_mood:
                    continue
            result.append(song)
            genre_counts[song.genre] = genre_counts.get(song.genre, 0) + 1
            last_mood = song.mood

        return result

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        contributions = {
            "valence":      0.30 * _gaussian(song.valence,      user.preferred_valence),
            "mood":         0.25 * (1.0 if song.mood == user.favorite_mood else 0.0),
            "energy":       0.20 * _gaussian(song.energy,       user.target_energy),
            "genre":        0.15 * (1.0 if song.genre == user.favorite_genre else 0.0),
            "acousticness": 0.10 * _gaussian(song.acousticness, user.preferred_acousticness),
        }
        top_feature = max(contributions, key=contributions.get)

        if top_feature == "mood":
            return f"Matched your favorite mood: {song.mood}"
        if top_feature == "genre":
            return f"Matched your favorite genre: {song.genre}"

        pref_map = {
            "valence":      user.preferred_valence,
            "energy":       user.target_energy,
            "acousticness": user.preferred_acousticness,
        }
        return (f"Matched your {top_feature} preference "
                f"({getattr(song, top_feature):.2f} vs your {pref_map[top_feature]:.2f})")


def load_songs(csv_path: str) -> List[Dict]:
    """
    Loads songs from a CSV file.
    Required by src/main.py
    """
    float_fields = {"energy", "valence", "acousticness", "danceability", "tempo_bpm"}
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["id"] = int(row["id"])
            for field in float_fields:
                row[field] = float(row[field])
            songs.append(row)
    return songs


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    Returns a list of (song_dict, score, explanation) tuples.
    """
    scored = sorted(songs, key=lambda s: _score_song_dict(s, user_prefs), reverse=True)

    result: List[Tuple[Dict, float, str]] = []
    genre_counts: Dict[str, int] = {}
    last_mood: str = ""

    for song in scored:
        if len(result) >= k:
            break
        if genre_counts.get(song["genre"], 0) >= 2:
            continue
        score = _score_song_dict(song, user_prefs)
        if result:
            last_score = result[-1][1]
            if abs(score - last_score) <= 0.05 and song["mood"] == last_mood:
                continue
        explanation = _explain_song_dict(song, user_prefs)
        result.append((song, score, explanation))
        genre_counts[song["genre"]] = genre_counts.get(song["genre"], 0) + 1
        last_mood = song["mood"]

    return result
