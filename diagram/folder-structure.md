# Folder Architecture

## System Data Flow

```
Streamlit UI  (src/app.py)
  ├── Step 1: Genre selectbox        (114 Kaggle genres)
  ├── Step 2: Mood selectbox         (10 moods)
  ├── Step 3: Energy slider          (0.0 → 1.0)
  ├── Step 4: Valence slider         (0.0 → 1.0)
  ├── Step 5: Acousticness slider    (0.0 → 1.0)
  └── "Get Recommendations" button
        │
        ▼
  src/rag.py  (RAG Retriever)
  ├── user_profile_to_vector(user_prefs)
  │     └── converts 5 prefs → 6-dim float32 query vector
  ├── infer_mood(energy, valence, tempo)
  │     └── derives mood label from audio features (10 mood classes)
  ├── retrieve_candidates(query_vec, matrix, meta, k=50)
  │     └── cosine similarity over 114k L2-normalized song vectors
  │         → returns top-50 candidate song dicts
  └── rag_recommend(user_prefs, index, meta, k_final=5)
        │  genre pre-filter → retrieve → deduplicate → re-rank
        ▼
  src/recommender.py  (Scoring Engine — unchanged from original Module 3 project)
  ├── _score_song_dict(song, user_prefs) → (score, [reasons])
  │     ├── genre_match    × 0.25   (binary: 1.0 exact match, 0.0 no match)
  │     ├── mood_match     × 0.25   (binary: 1.0 exact match, 0.0 no match)
  │     ├── valence        × 0.25   (Gaussian similarity, σ = 0.20)
  │     ├── energy         × 0.20   (Gaussian similarity, σ = 0.20)
  │     └── acousticness   × 0.05   (Gaussian similarity, σ = 0.20)
  ├── recommend_songs(songs, user_prefs, k) → sorted top-k with reasons
  └── Diversity filter (inside Recommender.recommend):
        ├── max 2 songs per genre in final results
        └── skip song if score within 0.05 of last pick AND same mood
        │
        ▼
  Streamlit results panel  (src/app.py)
  ├── Top 5 song cards    (title, artist, genre, mood, energy, valence)
  ├── Match score bar     (0.0 – 1.0 progress bar per song)
  ├── Per-song reason breakdown  (expandable: why each song was picked)
  └── "How the RAG works" expander  (educational step-by-step explanation)
```

---

## Supporting Components

```
data/
  ├── spotify_tracks.csv      114,000 Kaggle Spotify tracks — source data for RAG index
  │                           21 columns: track_name, artists, track_genre, energy,
  │                           valence, acousticness, danceability, tempo, …
  ├── rag_index.npz           L2-normalized 114k × 6 vector matrix
  │                           Auto-built on first Streamlit run (~5s); cached to disk
  ├── rag_index_meta.pkl      Song metadata dicts parallel to index rows
  │                           Auto-built alongside rag_index.npz; load time ~0.5s
  └── songs.csv               18-song hand-curated dataset
                              Used by src/main.py CLI harness for baseline evaluation

tests/
  ├── test_recommender.py     Song/UserProfile dataclasses, scoring edge cases,
  │                           OOP Recommender.recommend(), diversity filter, profiling
  └── test_rag.py             infer_mood() all 10 branches, vector shape/range check,
                              retrieve_candidates() returns k results, row key mapping

src/main.py                   CLI evaluation harness
                              Runs 4 predefined user profiles against songs.csv
                              Prints predicted vs. actual top songs and overlap count
```

---

## Directory Tree

```
ai110-module3show-musicrecommendersimulation-starter/
├── src/
│   ├── __init__.py
│   ├── app.py              Streamlit web UI
│   ├── rag.py              RAG retriever — vector index + cosine search
│   ├── recommender.py      Scoring engine — weighted formula + diversity filter
│   └── main.py             CLI evaluation harness
├── tests/
│   ├── test_recommender.py Unit tests for scoring logic
│   └── test_rag.py         Unit tests for RAG components
├── data/
│   ├── spotify_tracks.csv  114k Spotify tracks (Kaggle source)
│   ├── rag_index.npz       Pre-built vector index (auto-generated)
│   ├── rag_index_meta.pkl  Song metadata for index (auto-generated)
│   └── songs.csv           18-song baseline dataset
├── diagram/
│   ├── system-diagram.md   Mermaid flowchart of full RAG pipeline
│   └── folder-structure.md This file — data flow and codebase guide
├── README.md
├── model_card.md
├── requirements.txt
└── pytest.ini
```
