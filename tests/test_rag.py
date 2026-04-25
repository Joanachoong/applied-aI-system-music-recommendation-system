"""Unit tests for src/rag.py — no CSV or network access required."""

import numpy as np
import pytest

from src.rag import (
    infer_mood,
    song_vector,
    kaggle_row_to_song_dict,
    retrieve_candidates,
    user_profile_to_vector,
    rag_recommend,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_kaggle_row(**overrides) -> dict:
    base = {
        "Unnamed: 0": 1,
        "track_id": "abc123",
        "track_name": "Test Song",
        "artists": "Test Artist",
        "album_name": "Test Album",
        "track_genre": "pop",
        "energy": 0.80,
        "valence": 0.75,
        "acousticness": 0.10,
        "danceability": 0.70,
        "instrumentalness": 0.00,
        "tempo": 120.0,
        "popularity": 60,
        "duration_ms": 200000,
        "explicit": False,
        "key": 5,
        "loudness": -5.0,
        "mode": 1,
        "speechiness": 0.05,
        "liveness": 0.12,
        "time_signature": 4,
    }
    base.update(overrides)
    return base


def _make_small_index(n: int = 10):
    """Build a tiny in-memory index for retrieval tests."""
    rng = np.random.default_rng(42)
    matrix = rng.random((n, 6), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix_normed = matrix / np.clip(norms, 1e-8, None)
    song_dicts = [
        {
            "id": i, "title": f"Song {i}", "artist": "Artist",
            "genre": "pop", "mood": "happy",
            "energy": float(matrix[i, 0]), "valence": float(matrix[i, 1]),
            "acousticness": float(matrix[i, 2]), "danceability": float(matrix[i, 3]),
            "tempo_bpm": 120.0,
        }
        for i in range(n)
    ]
    return song_dicts, matrix_normed


# ── infer_mood ─────────────────────────────────────────────────────────────────

def test_infer_mood_energetic():
    assert infer_mood(valence=0.5, energy=0.90, acousticness=0.1, tempo=130) == "energetic"

def test_infer_mood_happy():
    assert infer_mood(valence=0.80, energy=0.70, acousticness=0.1, tempo=120) == "happy"

def test_infer_mood_intense():
    assert infer_mood(valence=0.30, energy=0.70, acousticness=0.1, tempo=140) == "intense"

def test_infer_mood_sad():
    assert infer_mood(valence=0.20, energy=0.25, acousticness=0.5, tempo=80) == "sad"

def test_infer_mood_chill():
    assert infer_mood(valence=0.60, energy=0.30, acousticness=0.5, tempo=90) == "chill"

def test_infer_mood_relaxed():
    # valence < 0.50 so "chill" rule doesn't fire; energy < 0.35 → "relaxed"
    assert infer_mood(valence=0.40, energy=0.25, acousticness=0.3, tempo=100) == "relaxed"

def test_infer_mood_fallback():
    result = infer_mood(valence=0.55, energy=0.55, acousticness=0.25, tempo=110)
    assert isinstance(result, str)
    assert len(result) > 0


# ── song_vector ────────────────────────────────────────────────────────────────

def test_song_vector_shape():
    row = _make_kaggle_row()
    vec = song_vector(row)
    assert vec.shape == (6,)

def test_song_vector_dtype():
    row = _make_kaggle_row()
    vec = song_vector(row)
    assert vec.dtype == np.float32

def test_song_vector_range():
    row = _make_kaggle_row(energy=0.8, valence=0.75, acousticness=0.1,
                           danceability=0.7, instrumentalness=0.0, tempo=120.0)
    vec = song_vector(row)
    assert np.all(vec >= 0.0) and np.all(vec <= 1.0)

def test_song_vector_tempo_normalised():
    row = _make_kaggle_row(tempo=250.0)
    vec = song_vector(row)
    assert vec[-1] == pytest.approx(1.0, abs=1e-4)


# ── kaggle_row_to_song_dict ────────────────────────────────────────────────────

REQUIRED_KEYS = {"id", "title", "artist", "genre", "mood",
                 "energy", "tempo_bpm", "valence", "danceability", "acousticness"}

def test_kaggle_row_to_song_dict_keys():
    row = _make_kaggle_row()
    sd = kaggle_row_to_song_dict(row)
    assert REQUIRED_KEYS.issubset(sd.keys())

def test_kaggle_row_to_song_dict_types():
    row = _make_kaggle_row()
    sd = kaggle_row_to_song_dict(row)
    assert isinstance(sd["id"], int)
    assert isinstance(sd["title"], str)
    assert isinstance(sd["energy"], float)

def test_kaggle_row_to_song_dict_mood_inferred():
    row = _make_kaggle_row(energy=0.90, valence=0.80)
    sd = kaggle_row_to_song_dict(row)
    assert sd["mood"] == "energetic"


# ── user_profile_to_vector ─────────────────────────────────────────────────────

def test_user_profile_to_vector_shape():
    prefs = {
        "target_energy": 0.8,
        "preferred_valence": 0.7,
        "preferred_acousticness": 0.1,
    }
    vec = user_profile_to_vector(prefs)
    assert vec.shape == (6,)

def test_user_profile_to_vector_range():
    prefs = {
        "target_energy": 0.8,
        "preferred_valence": 0.7,
        "preferred_acousticness": 0.1,
    }
    vec = user_profile_to_vector(prefs)
    assert np.all(vec >= 0.0) and np.all(vec <= 1.0)

def test_user_profile_to_vector_energy_mapped():
    prefs = {"target_energy": 0.9, "preferred_valence": 0.5, "preferred_acousticness": 0.2}
    vec = user_profile_to_vector(prefs)
    assert vec[0] == pytest.approx(0.9, abs=1e-4)


# ── retrieve_candidates ────────────────────────────────────────────────────────

def test_retrieve_returns_k_results():
    song_dicts, matrix_normed = _make_small_index(n=10)
    query = np.array([0.8, 0.7, 0.1, 0.6, 0.0, 0.5], dtype=np.float32)
    results = retrieve_candidates(query, song_dicts, matrix_normed, k=5)
    assert len(results) == 5

def test_retrieve_returns_all_when_k_exceeds_n():
    song_dicts, matrix_normed = _make_small_index(n=4)
    query = np.array([0.8, 0.7, 0.1, 0.6, 0.0, 0.5], dtype=np.float32)
    results = retrieve_candidates(query, song_dicts, matrix_normed, k=10)
    assert len(results) == 4

def test_retrieve_returns_dicts_with_required_keys():
    song_dicts, matrix_normed = _make_small_index(n=6)
    query = np.array([0.5, 0.5, 0.3, 0.5, 0.1, 0.5], dtype=np.float32)
    results = retrieve_candidates(query, song_dicts, matrix_normed, k=3)
    for r in results:
        assert "title" in r and "genre" in r and "energy" in r


# ── rag_recommend ──────────────────────────────────────────────────────────────

def test_rag_recommend_returns_k():
    song_dicts, matrix_normed = _make_small_index(n=10)
    user_prefs = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.8,
        "preferred_valence": 0.7,
        "preferred_acousticness": 0.1,
    }
    results = rag_recommend(user_prefs, song_dicts, matrix_normed, k_retrieve=8, k_final=3)
    assert len(results) <= 3

def test_rag_recommend_result_structure():
    song_dicts, matrix_normed = _make_small_index(n=10)
    user_prefs = {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.8,
        "preferred_valence": 0.7,
        "preferred_acousticness": 0.1,
    }
    results = rag_recommend(user_prefs, song_dicts, matrix_normed, k_retrieve=8, k_final=3)
    for song, score, reasons in results:
        assert isinstance(song, dict)
        assert 0.0 <= score <= 1.0
        assert isinstance(reasons, list)
