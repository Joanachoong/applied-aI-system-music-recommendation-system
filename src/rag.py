"""
RAG retrieval layer for the music recommender.

Builds a 6-dim audio-feature vector index over the 114k Kaggle Spotify dataset,
then retrieves the top-k sonically similar songs for a given user profile. The
retrieved candidates are fed into the existing _score_song_dict / recommend_songs
pipeline unchanged.
"""

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.recommender import recommend_songs

# ── Embedding column order (must stay consistent between build & query) ────────
_EMBED_COLS = ["energy", "valence", "acousticness", "danceability", "instrumentalness"]
# tempo is a 6th feature, normalised separately (max ~250 BPM)
_TEMPO_MAX = 250.0


# ── Mood inference ─────────────────────────────────────────────────────────────

def infer_mood(valence: float, energy: float, acousticness: float, tempo: float) -> str:
    """Derive a mood label from Kaggle audio features (first match wins)."""
    if energy >= 0.80:
        return "energetic"
    if valence >= 0.70 and energy >= 0.60:
        return "happy"
    if valence < 0.45 and energy >= 0.65:
        return "intense"
    if valence < 0.40 and energy < 0.45:
        return "sad"
    if valence >= 0.50 and energy < 0.40:
        return "chill"
    if energy < 0.35:
        return "relaxed"
    if 0.30 <= energy <= 0.60 and acousticness < 0.30:
        return "focused"
    if acousticness >= 0.50 and 0.40 <= valence <= 0.70:
        return "romantic"
    if tempo < 95 and acousticness >= 0.40:
        return "nostalgic"
    return "moody"


# ── Vector construction ────────────────────────────────────────────────────────

def song_vector(row: Dict) -> np.ndarray:
    """Convert a Kaggle track dict to a 6-dim float32 feature vector."""
    vec = np.array(
        [float(row[c]) for c in _EMBED_COLS] + [float(row["tempo"]) / _TEMPO_MAX],
        dtype=np.float32,
    )
    return np.clip(vec, 0.0, 1.0)


def kaggle_row_to_song_dict(row: Dict) -> Dict:
    """Map a Kaggle CSV row to the recommender song-dict schema."""
    valence = float(row["valence"])
    energy = float(row["energy"])
    acousticness = float(row["acousticness"])
    tempo = float(row["tempo"])
    return {
        "id": int(row["Unnamed: 0"]),
        "title": str(row["track_name"]),
        "artist": str(row["artists"]),
        "genre": str(row["track_genre"]),
        "mood": infer_mood(valence, energy, acousticness, tempo),
        "energy": energy,
        "tempo_bpm": tempo,
        "valence": valence,
        "danceability": float(row["danceability"]),
        "acousticness": acousticness,
    }


# ── Index build & load ─────────────────────────────────────────────────────────

def load_kaggle_tracks(csv_path: str) -> Tuple[List[Dict], np.ndarray]:
    """Load the Kaggle CSV, convert rows, and return (song_dicts, raw_matrix)."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["track_name", "artists", "track_genre",
                            "energy", "valence", "acousticness",
                            "danceability", "instrumentalness", "tempo"])

    song_dicts: List[Dict] = []
    vectors: List[np.ndarray] = []
    for _, row in df.iterrows():
        try:
            sd = kaggle_row_to_song_dict(row.to_dict())
            vec = song_vector(row.to_dict())
        except (ValueError, KeyError):
            continue
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            continue
        song_dicts.append(sd)
        vectors.append(vec)

    matrix = np.stack(vectors, axis=0).astype(np.float32)  # (N, 6)
    return song_dicts, matrix


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-8, None)


def build_index(csv_path: str, index_path: str) -> None:
    """Build the L2-normalised vector index and persist it to disk."""
    print("Building RAG index from Kaggle dataset…")
    song_dicts, matrix = load_kaggle_tracks(csv_path)
    matrix_normed = _normalize(matrix)

    np.savez_compressed(index_path, matrix_normed=matrix_normed)
    meta_path = index_path.replace(".npz", "_meta.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(song_dicts, f)
    print(f"Index built: {len(song_dicts):,} songs → {index_path}")


def load_index(index_path: str) -> Tuple[List[Dict], np.ndarray]:
    """Load a pre-built index from disk."""
    data = np.load(index_path)
    matrix_normed = data["matrix_normed"]
    meta_path = index_path.replace(".npz", "_meta.pkl")
    with open(meta_path, "rb") as f:
        song_dicts = pickle.load(f)
    return song_dicts, matrix_normed


def get_or_build_index(
    csv_path: str = "data/spotify_tracks.csv",
    index_path: str = "data/rag_index.npz",
) -> Tuple[List[Dict], np.ndarray]:
    """Return the cached index if it exists, otherwise build and cache it."""
    if os.path.exists(index_path):
        return load_index(index_path)
    build_index(csv_path, index_path)
    return load_index(index_path)


# ── Retrieval ──────────────────────────────────────────────────────────────────

def user_profile_to_vector(user_prefs: Dict) -> np.ndarray:
    """Convert a user_prefs dict to a 6-dim embedding vector."""
    vec = np.array(
        [
            float(user_prefs.get("target_energy", 0.5)),
            float(user_prefs.get("preferred_valence", 0.5)),
            float(user_prefs.get("preferred_acousticness", 0.3)),
            float(user_prefs.get("preferred_danceability", 0.5)),
            float(user_prefs.get("preferred_instrumentalness", 0.1)),
            float(user_prefs.get("preferred_tempo_bpm", 120.0)) / _TEMPO_MAX,
        ],
        dtype=np.float32,
    )
    return np.clip(vec, 0.0, 1.0)


def retrieve_candidates(
    query_vec: np.ndarray,
    song_dicts: List[Dict],
    matrix_normed: np.ndarray,
    k: int = 50,
) -> List[Dict]:
    """Return the top-k song dicts by cosine similarity to query_vec."""
    norm = float(np.linalg.norm(query_vec))
    if norm < 1e-8:
        return song_dicts[:k]
    query_normed = query_vec / norm
    scores = matrix_normed @ query_normed  # (N,)
    k_clamped = min(k, len(song_dicts))
    top_indices = np.argpartition(scores, -k_clamped)[-k_clamped:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return [song_dicts[i] for i in top_indices]


def rag_recommend(
    user_prefs: Dict,
    song_dicts: List[Dict],
    matrix_normed: np.ndarray,
    k_retrieve: int = 50,
    k_final: int = 5,
) -> List[Tuple[Dict, float, List[str]]]:
    """
    Full RAG pipeline:
      1. Convert user_prefs to a 6-dim query vector
      2. Genre pre-filter: restrict to songs matching the user's favorite_genre
         (falls back to full corpus if fewer than k_retrieve genre matches exist)
      3. Cosine-retrieve top-k_retrieve candidates from the filtered set
      4. Re-rank with the existing recommend_songs / _score_song_dict engine
    Returns List[(song_dict, score, reasons)].
    """
    query_vec = user_profile_to_vector(user_prefs)
    genre = user_prefs.get("favorite_genre", "")

    # Build a genre-filtered sub-index when there are enough matching songs
    if genre:
        genre_indices = [i for i, s in enumerate(song_dicts) if s["genre"] == genre]
    else:
        genre_indices = []

    if len(genre_indices) >= k_retrieve:
        filtered_dicts = [song_dicts[i] for i in genre_indices]
        filtered_matrix = matrix_normed[np.array(genre_indices)]
    else:
        # Not enough genre matches — search the full corpus (genre bonus still applies in scorer)
        filtered_dicts = song_dicts
        filtered_matrix = matrix_normed

    candidates = retrieve_candidates(query_vec, filtered_dicts, filtered_matrix, k=k_retrieve)
    if not candidates:
        return []

    # Deduplicate by (title, artist) — Kaggle has multiple entries per song
    seen: set = set()
    unique_candidates: List[Dict] = []
    for c in candidates:
        key = (c["title"].lower(), c["artist"].lower())
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)

    return recommend_songs(user_prefs, unique_candidates, k=k_final)
