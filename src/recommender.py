"""Recommendation engine for the project, including dataclasses, scoring, and a working CLI-first simulation."""

import csv
import math
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class ScoringWeights:
    """Per-feature weights for a scoring strategy. All fields should sum to 1.0."""
    genre: float
    mood: float
    valence: float
    energy: float
    acousticness: float
    artist: float


SCORING_MODES: Dict[str, ScoringWeights] = {
    "Balanced":       ScoringWeights(genre=0.25, mood=0.25, valence=0.25, energy=0.20, acousticness=0.05, artist=0.00),
    "Genre-First":    ScoringWeights(genre=0.50, mood=0.20, valence=0.15, energy=0.10, acousticness=0.05, artist=0.00),
    "Mood-First":     ScoringWeights(genre=0.20, mood=0.50, valence=0.15, energy=0.10, acousticness=0.05, artist=0.00),
    # Artist-Match: artist holds 0.50; remaining 0.50 uses Balanced proportions.
    "Artist-Match":   ScoringWeights(genre=0.125, mood=0.125, valence=0.125, energy=0.10, acousticness=0.025, artist=0.50),
}


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
    popularity: float = 50.0

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
    preferred_artist: str = ""


def _gaussian(song_val: float, pref_val: float, sigma: float = 0.20) -> float:
    """Gaussian similarity: 1.0 when values match, exponentially lower as they diverge."""
    return math.exp(-((song_val - pref_val) ** 2) / (2 * sigma ** 2))


def _score_song_dict(song: Dict, user_prefs: Dict, weights: ScoringWeights = None) -> Tuple[float, List[str]]:
    """Score a song dict against a user_prefs dict.

    Returns:
        (score, reasons) where score is in [0, 1] and reasons is a list of
        human-readable strings explaining each feature's contribution.
        Every feature always produces a reason so the caller sees a full breakdown.
    """
    if weights is None:
        weights = SCORING_MODES["Balanced"]

    reasons: List[str] = []

    # --- categorical features: binary match/mismatch ---
    if song["genre"] == user_prefs["favorite_genre"]:
        genre_contribution = weights.genre
        reasons.append(f"genre match (+{genre_contribution:.2f})")
    else:
        genre_contribution = 0.0
        reasons.append(
            f"genre mismatch (+0.00, song={song['genre']}, "
            f"you prefer={user_prefs['favorite_genre']})"
        )

    if song["mood"] == user_prefs["favorite_mood"]:
        mood_contribution = weights.mood
        reasons.append(f"mood match (+{mood_contribution:.2f})")
    else:
        mood_contribution = 0.0
        reasons.append(
            f"mood mismatch (+0.00, song={song['mood']}, "
            f"you prefer={user_prefs['favorite_mood']})"
        )

    # --- numerical features: Gaussian similarity (always contributes something) ---
    valence_contribution  = weights.valence   * _gaussian(song["valence"],      user_prefs["preferred_valence"])
    energy_contribution   = weights.energy    * _gaussian(song["energy"],       user_prefs["target_energy"])
    acoustic_contribution = weights.acousticness * _gaussian(song["acousticness"], user_prefs["preferred_acousticness"])

    reasons.append(
        f"valence similarity (+{valence_contribution:.2f}, "
        f"song={song['valence']:.2f} vs your {user_prefs['preferred_valence']:.2f})"
    )
    reasons.append(
        f"energy similarity (+{energy_contribution:.2f}, "
        f"song={song['energy']:.2f} vs your {user_prefs['target_energy']:.2f})"
    )
    reasons.append(
        f"acousticness similarity (+{acoustic_contribution:.2f}, "
        f"song={song['acousticness']:.2f} vs your {user_prefs['preferred_acousticness']:.2f})"
    )

    # --- artist feature: popularity-weighted, only active when weights.artist > 0 ---
    artist_contribution = 0.0
    if weights.artist > 0:
        preferred_artist = user_prefs.get("preferred_artist", "")
        pop = min(float(song.get("popularity", 50)), 100.0)
        if preferred_artist and song["artist"].strip().lower() == preferred_artist.strip().lower():
            artist_contribution = weights.artist * (pop / 100.0)
            reasons.append(
                f"artist match (+{artist_contribution:.2f}, {song['artist']}, popularity={int(pop)}/100)"
            )
        elif preferred_artist:
            reasons.append(
                f"artist mismatch (+0.00, song={song['artist']}, you prefer={preferred_artist})"
            )
        else:
            reasons.append("artist (skipped — no preferred artist set)")

    score = (genre_contribution + mood_contribution
             + valence_contribution + energy_contribution + acoustic_contribution
             + artist_contribution)

    return score, reasons


def _score_song_obj(song: "Song", user: "UserProfile", weights: ScoringWeights = None) -> Tuple[float, List[str]]:
    """Score a Song dataclass against a UserProfile dataclass.

    Returns:
        (score, reasons) — same contract as _score_song_dict(), using object
        attributes instead of dict keys.
    """
    if weights is None:
        weights = SCORING_MODES["Balanced"]

    reasons: List[str] = []

    # --- categorical features: binary match/mismatch ---
    if song.genre == user.favorite_genre:
        genre_contribution = weights.genre
        reasons.append(f"genre match (+{genre_contribution:.2f})")
    else:
        genre_contribution = 0.0
        reasons.append(
            f"genre mismatch (+0.00, song={song.genre}, "
            f"you prefer={user.favorite_genre})"
        )

    if song.mood == user.favorite_mood:
        mood_contribution = weights.mood
        reasons.append(f"mood match (+{mood_contribution:.2f})")
    else:
        mood_contribution = 0.0
        reasons.append(
            f"mood mismatch (+0.00, song={song.mood}, "
            f"you prefer={user.favorite_mood})"
        )

    # --- numerical features: Gaussian similarity ---
    valence_contribution  = weights.valence      * _gaussian(song.valence,      user.preferred_valence)
    energy_contribution   = weights.energy       * _gaussian(song.energy,       user.target_energy)
    acoustic_contribution = weights.acousticness * _gaussian(song.acousticness, user.preferred_acousticness)

    reasons.append(
        f"valence similarity (+{valence_contribution:.2f}, "
        f"song={song.valence:.2f} vs your {user.preferred_valence:.2f})"
    )
    reasons.append(
        f"energy similarity (+{energy_contribution:.2f}, "
        f"song={song.energy:.2f} vs your {user.target_energy:.2f})"
    )
    reasons.append(
        f"acousticness similarity (+{acoustic_contribution:.2f}, "
        f"song={song.acousticness:.2f} vs your {user.preferred_acousticness:.2f})"
    )

    # --- artist feature: popularity-weighted, only active when weights.artist > 0 ---
    artist_contribution = 0.0
    if weights.artist > 0:
        preferred_artist = user.preferred_artist
        pop = min(float(getattr(song, "popularity", 50.0)), 100.0)
        if preferred_artist and song.artist.strip().lower() == preferred_artist.strip().lower():
            artist_contribution = weights.artist * (pop / 100.0)
            reasons.append(
                f"artist match (+{artist_contribution:.2f}, {song.artist}, popularity={int(pop)}/100)"
            )
        elif preferred_artist:
            reasons.append(
                f"artist mismatch (+0.00, song={song.artist}, you prefer={preferred_artist})"
            )
        else:
            reasons.append("artist (skipped — no preferred artist set)")

    score = (genre_contribution + mood_contribution
             + valence_contribution + energy_contribution + acoustic_contribution
             + artist_contribution)

    return score, reasons



class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5, mode: str = "Balanced") -> List[Song]:
        """Return the top-k songs for a user profile."""
        weights = SCORING_MODES.get(mode, SCORING_MODES["Balanced"])

        # Artist-Match fast path: return ALL songs by the preferred artist, ranked by popularity
        if mode == "Artist-Match" and user.preferred_artist:
            preferred = user.preferred_artist.strip().lower()
            artist_songs = [s for s in self.songs if s.artist.strip().lower() == preferred]
            return sorted(artist_songs, key=lambda s: float(getattr(s, "popularity", 50)), reverse=True)

        # Pre-compute scores once to avoid double-scoring in the diversity loop
        pre_scored = [(s, _score_song_obj(s, user, weights)[0]) for s in self.songs]
        scored = [s for s, _ in sorted(pre_scored, key=lambda x: x[1], reverse=True)]

        result: List[Song] = []
        last_score: float = 0.0
        genre_counts: Dict[str, int] = {}
        last_mood: str = ""

        for song in scored:
            if len(result) >= k:
                break
            if genre_counts.get(song.genre, 0) >= 2:
                continue
            this_score, _ = _score_song_obj(song, user, weights)
            # If score is within 0.05 of the last added song, prefer a different mood
            if result and abs(this_score - last_score) <= 0.05 and song.mood == last_mood:
                continue
            result.append(song)
            last_score = this_score
            genre_counts[song.genre] = genre_counts.get(song.genre, 0) + 1
            last_mood = song.mood

        return result

    def explain_recommendation(self, user: UserProfile, song: Song, mode: str = "Balanced") -> str:
        """Explain the strongest matching feature for a song."""
        weights = SCORING_MODES.get(mode, SCORING_MODES["Balanced"])
        pop = min(float(getattr(song, "popularity", 50.0)), 100.0)
        preferred_artist = user.preferred_artist

        contributions = {
            "valence":      weights.valence      * _gaussian(song.valence,      user.preferred_valence),
            "mood":         weights.mood         * (1.0 if song.mood == user.favorite_mood else 0.0),
            "energy":       weights.energy       * _gaussian(song.energy,       user.target_energy),
            "genre":        weights.genre        * (1.0 if song.genre == user.favorite_genre else 0.0),
            "acousticness": weights.acousticness * _gaussian(song.acousticness, user.preferred_acousticness),
            "artist":       weights.artist * (pop / 100.0) if (
                weights.artist > 0 and preferred_artist and
                song.artist.strip().lower() == preferred_artist.strip().lower()
            ) else 0.0,
        }
        top_feature = max(contributions, key=contributions.get)

        if top_feature == "mood":
            return f"Matched your favorite mood: {song.mood}"
        if top_feature == "genre":
            return f"Matched your favorite genre: {song.genre}"
        if top_feature == "artist":
            return f"Matched your preferred artist: {song.artist} (popularity {int(pop)}/100)"

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


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5, mode: str = "Balanced") -> List[Tuple[Dict, float, List[str]]]:
    """
    Functional implementation of the recommendation logic.
    Required by src/main.py
    Returns a list of (song_dict, score, reasons) tuples where reasons is a
    list of human-readable strings explaining each feature's contribution.
    """
    weights = SCORING_MODES.get(mode, SCORING_MODES["Balanced"])

    # Artist-Match fast path: artist overrides all other features.
    # Score = popularity/100; genre/mood/energy are ignored.
    if mode == "Artist-Match" and user_prefs.get("preferred_artist"):
        preferred = user_prefs["preferred_artist"].strip().lower()
        artist_songs = [s for s in songs if s["artist"].strip().lower() == preferred]
        scored = []
        for s in artist_songs:
            pop = min(float(s.get("popularity", 50)), 100.0)
            score = pop / 100.0
            reasons = [f"artist match ({s['artist']}), popularity={int(pop)}/100"]
            scored.append((s, score, reasons))
        return sorted(scored, key=lambda x: x[1], reverse=True)[:k]

    # Score every song once
    scored = [
        (song, score, reasons)
        for song in songs
        for score, reasons in [_score_song_dict(song, user_prefs, weights)]
    ]

    # Sort by score (highest first), then slice to the top k
    return sorted(scored, key=lambda x: x[1], reverse=True)[:k]
