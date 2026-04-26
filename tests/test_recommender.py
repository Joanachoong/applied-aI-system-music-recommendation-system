"""Runtime-behavior tests for src/recommender.py.

Layer 1 — Direct _score_song_dict() scoring verification:
  Calls the scoring function with each mode's exact ScoringWeights and asserts
  the expected score, confirming weight tables are applied correctly.

Layer 2 — End-to-end recommend_songs() behavior tests:
  Uses a controlled 18-song fixture to mirror real Streamlit user interactions.
  All top-5 match scores are verified against the > 0.7 quality threshold shown
  in the UI (app.py renders the raw score via st.metric / st.progress).
"""

import pytest
from src.recommender import (
    _score_song_dict,
    recommend_songs,
    SCORING_MODES,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

# Perfect-match song for Layer 1 scoring unit tests.
# All numerical values equal the user preference → all Gaussian terms = 1.0.
_PERFECT_SONG = {
    "genre": "pop", "mood": "happy",
    "energy": 0.80, "valence": 0.75, "acousticness": 0.15,
    "artist": "PopStar", "popularity": 90,
}

_USER_PREFS = {
    "favorite_genre": "pop", "favorite_mood": "happy",
    "target_energy": 0.80, "preferred_valence": 0.75,
    "preferred_acousticness": 0.15, "preferred_artist": "PopStar",
}


def _make_rich_fixture() -> list:
    """18-song controlled dataset for Layer 2 behavioral tests.

    Group A (ids 1-5):  pop/happy, energy ≈ 0.80, artist=PopStar, popularity 90→55
    Group B (ids 6-10): varied genres (rock/electronic/hip-hop/r-n-b/indie), energy ≈ 0.80
    Group C (ids 11-12): pop/sad, low energy — for conflict tests
    Group D (id 13):    rock/sad noise
    Group E (ids 14-15): very low energy — contrast for Energy-Focused test
    Group F (ids 16-18): artist=RareArtist (only 3 songs) — for edge-case test
    """
    return [
        # Group A
        {"id": 1,  "title": "Pop Happy Anthem",  "artist": "PopStar",    "genre": "pop",        "mood": "happy",     "energy": 0.80, "valence": 0.75, "acousticness": 0.15, "popularity": 90},
        {"id": 2,  "title": "Pop Vibes",          "artist": "PopStar",    "genre": "pop",        "mood": "happy",     "energy": 0.79, "valence": 0.74, "acousticness": 0.16, "popularity": 80},
        {"id": 3,  "title": "Pop Sunshine",       "artist": "PopStar",    "genre": "pop",        "mood": "happy",     "energy": 0.81, "valence": 0.76, "acousticness": 0.14, "popularity": 70},
        {"id": 4,  "title": "Pop Summer Hit",     "artist": "PopStar",    "genre": "pop",        "mood": "happy",     "energy": 0.78, "valence": 0.73, "acousticness": 0.17, "popularity": 65},
        {"id": 5,  "title": "Pop Dance Floor",    "artist": "PopStar",    "genre": "pop",        "mood": "happy",     "energy": 0.82, "valence": 0.77, "acousticness": 0.13, "popularity": 55},
        # Group B
        {"id": 6,  "title": "Rock Power",         "artist": "RockBand",   "genre": "rock",       "mood": "energetic", "energy": 0.82, "valence": 0.55, "acousticness": 0.10, "popularity": 75},
        {"id": 7,  "title": "Electronic Surge",   "artist": "DJPro",      "genre": "electronic", "mood": "happy",     "energy": 0.80, "valence": 0.70, "acousticness": 0.05, "popularity": 70},
        {"id": 8,  "title": "HipHop Heat",        "artist": "Rapper1",    "genre": "hip-hop",    "mood": "energetic", "energy": 0.79, "valence": 0.60, "acousticness": 0.08, "popularity": 80},
        {"id": 9,  "title": "RnB Feel",           "artist": "RnBStar",    "genre": "r-n-b",      "mood": "happy",     "energy": 0.81, "valence": 0.65, "acousticness": 0.12, "popularity": 55},
        {"id": 10, "title": "Indie Drive",        "artist": "IndieAct",   "genre": "indie",      "mood": "energetic", "energy": 0.78, "valence": 0.58, "acousticness": 0.20, "popularity": 50},
        # Group C
        {"id": 11, "title": "Pop Melancholy",     "artist": "SadArtist",  "genre": "pop",        "mood": "sad",       "energy": 0.35, "valence": 0.25, "acousticness": 0.50, "popularity": 55},
        {"id": 12, "title": "Pop Heartbreak",     "artist": "SadArtist2", "genre": "pop",        "mood": "sad",       "energy": 0.30, "valence": 0.20, "acousticness": 0.55, "popularity": 50},
        # Group D
        {"id": 13, "title": "Rock Lament",        "artist": "SadRocker",  "genre": "rock",       "mood": "sad",       "energy": 0.30, "valence": 0.25, "acousticness": 0.55, "popularity": 45},
        # Group E
        {"id": 14, "title": "Classical Serenity", "artist": "Composer",   "genre": "classical",  "mood": "relaxed",   "energy": 0.20, "valence": 0.40, "acousticness": 0.80, "popularity": 30},
        {"id": 15, "title": "Jazz Lounge",        "artist": "JazzMan",    "genre": "jazz",       "mood": "chill",     "energy": 0.35, "valence": 0.55, "acousticness": 0.60, "popularity": 35},
        # Group F
        {"id": 16, "title": "Rare Pop Jam",       "artist": "RareArtist", "genre": "pop",        "mood": "happy",     "energy": 0.80, "valence": 0.75, "acousticness": 0.15, "popularity": 85},
        {"id": 17, "title": "Rare Pop Groove",    "artist": "RareArtist", "genre": "pop",        "mood": "happy",     "energy": 0.79, "valence": 0.74, "acousticness": 0.16, "popularity": 72},
        {"id": 18, "title": "Rare Pop Echo",      "artist": "RareArtist", "genre": "pop",        "mood": "happy",     "energy": 0.81, "valence": 0.76, "acousticness": 0.14, "popularity": 60},
    ]


_DEFAULT_PREFS = {
    "favorite_genre": "pop",
    "favorite_mood": "happy",
    "target_energy": 0.80,
    "preferred_valence": 0.75,
    "preferred_acousticness": 0.15,
    "preferred_artist": "",
}


# ── Layer 1: _score_song_dict() weight verification ───────────────────────────

def test_score_balanced_perfect_match():
    """Balanced: genre(0.25)+mood(0.25)+valence(0.25)+energy(0.20)+acousticness(0.05) = 1.0."""
    score, _ = _score_song_dict(_PERFECT_SONG, _USER_PREFS, SCORING_MODES["Balanced"])
    assert score == pytest.approx(1.0, abs=1e-4)


def test_score_genre_first_perfect_match():
    """Genre-First: genre(0.50)+mood(0.20)+valence(0.15)+energy(0.10)+acousticness(0.05) = 1.0."""
    score, _ = _score_song_dict(_PERFECT_SONG, _USER_PREFS, SCORING_MODES["Genre-First"])
    assert score == pytest.approx(1.0, abs=1e-4)


def test_score_mood_first_perfect_match():
    """Mood-First: genre(0.20)+mood(0.50)+valence(0.15)+energy(0.10)+acousticness(0.05) = 1.0."""
    score, _ = _score_song_dict(_PERFECT_SONG, _USER_PREFS, SCORING_MODES["Mood-First"])
    assert score == pytest.approx(1.0, abs=1e-4)


def test_score_artist_match_perfect_match():
    """Artist-Match: artist(0.50×90/100=0.45) + genre(0.125)+mood(0.125)+valence(0.125)
    +energy(0.10)+acousticness(0.025) = 0.95."""
    score, _ = _score_song_dict(_PERFECT_SONG, _USER_PREFS, SCORING_MODES["Artist-Match"])
    assert score == pytest.approx(0.95, abs=1e-4)


def test_score_genre_first_genre_mismatch():
    """Genre-First genre mismatch: genre weight 0.50 drops to 0.
    song(rock/happy) vs user(pop/happy): 0+mood(0.20)+valence(0.15)+energy(0.10)+acousticness(0.05) = 0.50."""
    mismatched = {**_PERFECT_SONG, "genre": "rock"}
    score, _ = _score_song_dict(mismatched, _USER_PREFS, SCORING_MODES["Genre-First"])
    assert score == pytest.approx(0.50, abs=1e-4)


def test_score_mood_first_mood_mismatch():
    """Mood-First mood mismatch: mood weight 0.50 drops to 0.
    song(pop/sad) vs user(pop/happy): genre(0.20)+0+valence(0.15)+energy(0.10)+acousticness(0.05) = 0.50."""
    mismatched = {**_PERFECT_SONG, "mood": "sad"}
    score, _ = _score_song_dict(mismatched, _USER_PREFS, SCORING_MODES["Mood-First"])
    assert score == pytest.approx(0.50, abs=1e-4)


# ── Layer 2: recommend_songs() runtime behavior tests ─────────────────────────

def test_balanced_all_scores_above_threshold():
    """Balanced mode: top-5 songs all score > 0.7 — the quality bar shown in the UI."""
    results = recommend_songs(_DEFAULT_PREFS, _make_rich_fixture(), k=5, mode="Balanced")
    assert len(results) == 5
    for song, score, _ in results:
        assert score > 0.7, f"{song['title']} scored {score:.3f}, expected > 0.7"


def test_genre_first_top5_same_genre():
    """Genre-First: genre weight 0.50 dominates; all top-5 results should be 'pop'."""
    results = recommend_songs(_DEFAULT_PREFS, _make_rich_fixture(), k=5, mode="Genre-First")
    assert len(results) == 5
    for song, score, _ in results:
        assert song["genre"] == "pop", f"Expected pop, got {song['genre']}"
        assert score > 0.7, f"{song['title']} scored {score:.3f}, expected > 0.7"


def test_mood_first_top5_same_mood():
    """Mood-First: mood weight 0.50 dominates; all top-5 results should be 'happy'."""
    results = recommend_songs(_DEFAULT_PREFS, _make_rich_fixture(), k=5, mode="Mood-First")
    assert len(results) == 5
    for song, score, _ in results:
        assert song["mood"] == "happy", f"Expected happy, got {song['mood']}"
        assert score > 0.7, f"{song['title']} scored {score:.3f}, expected > 0.7"


def test_conflict_genre_vs_mood():
    """User selects genre=pop but mood=sad — a common Streamlit conflict pattern.

    Genre-First (genre 0.50): pop/happy scores 0.80 vs pop/sad 0.726 → happy wins.
    Mood-First  (mood  0.50): pop/sad  scores 0.726 vs pop/happy 0.50 → sad wins.
    The same preferences produce different top results depending on scoring mode.
    """
    conflict_prefs = {**_DEFAULT_PREFS, "favorite_mood": "sad"}
    songs = _make_rich_fixture()

    genre_results = recommend_songs(conflict_prefs, songs, k=5, mode="Genre-First")
    mood_results  = recommend_songs(conflict_prefs, songs, k=5, mode="Mood-First")

    assert genre_results[0][0]["mood"] != "sad", (
        f"Genre-First should rank pop/happy first; got mood={genre_results[0][0]['mood']}"
    )
    assert mood_results[0][0]["mood"] == "sad", (
        f"Mood-First should rank pop/sad first; got mood={mood_results[0][0]['mood']}"
    )


def test_artist_match_top5_by_preferred_artist_sorted_by_popularity():
    """Artist-Match: all 5 results should be by 'PopStar', ordered by decreasing
    popularity (artist contribution = 0.50 × popularity/100 drives the ranking).
    """
    prefs = {**_DEFAULT_PREFS, "preferred_artist": "PopStar"}
    results = recommend_songs(prefs, _make_rich_fixture(), k=5, mode="Artist-Match")
    assert len(results) == 5
    for song, score, _ in results:
        assert song["artist"] == "PopStar", f"Expected PopStar, got {song['artist']}"
        assert score > 0.7, f"{song['title']} scored {score:.3f}, expected > 0.7"
    popularities = [song["popularity"] for song, _, _ in results]
    assert popularities == sorted(popularities, reverse=True), (
        f"Results not sorted by popularity: {popularities}"
    )


def test_artist_match_edge_case_fewer_than_5_songs():
    """Artist-Match edge case: preferred artist has only 3 songs in the dataset.
    Top 3 should be the artist's songs (score > 0.7); positions 4-5 fall back to
    songs with a similar profile (pop/happy) since they score highest after artist songs.
    """
    prefs = {**_DEFAULT_PREFS, "preferred_artist": "RareArtist"}
    results = recommend_songs(prefs, _make_rich_fixture(), k=5, mode="Artist-Match")
    assert len(results) == 5

    for song, score, _ in results[:3]:
        assert song["artist"] == "RareArtist", (
            f"Expected RareArtist in top 3, got {song['artist']}"
        )
        assert score > 0.7, f"{song['title']} scored {score:.3f}, expected > 0.7"

    for song, _, _ in results[3:]:
        assert song["artist"] != "RareArtist", (
            "Only 3 RareArtist songs exist; positions 4-5 must be fallback songs"
        )
        assert song["genre"] == "pop", f"Fallback should be pop genre, got {song['genre']}"
        assert song["mood"] == "happy", f"Fallback should be happy mood, got {song['mood']}"
