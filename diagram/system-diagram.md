## Complete System Diagram

The diagram below shows the **extended RAG architecture** that upgrades the original 18-song system to search across 114,000 real Spotify tracks. The existing scoring engine (`_score_song_dict`, `recommend_songs`) is unchanged — RAG adds a retrieval step *in front of* it.

```mermaid
flowchart TD
    USER(["👤 User"])

    subgraph UI["Streamlit UI  (src/app.py)"]
        FORM["Preference Form\n─────────────\nGenre selectbox (114 genres)\nMood selectbox (10 moods)\nEnergy slider\nValence slider\nAcousticness slider"]
        RESULTS["Results Panel\n─────────────\nTop-5 song cards\nScore bars\nReason breakdown\nRAG explainer"]
    end

    subgraph RAG["RAG Retriever  (src/rag.py)"]
        VECTORIZE["user_profile_to_vector()\n──────────────────────\nConvert prefs → 6-dim\nquery vector"]
        COSINE["retrieve_candidates()\n──────────────────────\nCosine similarity vs\n114k song vectors\n→ top-50 candidates"]
        INFER["infer_mood()\n──────────────────────\nDerive mood label from\nvalence + energy + tempo"]
    end

    subgraph INDEX["Vector Index  (data/)"]
        KAGGLE[("📀 spotify_tracks.csv\n114,000 songs\n21 Kaggle columns")]
        MATRIX[("🗄️ rag_index.npz\nL2-normalized\n114k × 6 matrix\n+ metadata pickle")]
    end

    subgraph SCORE["Scoring Engine  (src/recommender.py)  — unchanged"]
        RERANK["recommend_songs()\n_score_song_dict()\n──────────────────────\nGenre   +0.25\nMood    +0.25\nValence +0.25 × gaussian\nEnergy  +0.20 × gaussian\nAcoustic+0.05 × gaussian"]
    end

    subgraph TESTS["Test Suite  (tests/)"]
        UNIT["test_rag.py\n──────────────────\ninfer_mood branches\nvector shape / range\nretrieve returns k\nrow mapping keys"]
        HUMAN["👤 Manual Review\n──────────────────\nDev runs Streamlit\nverifies K-pop query\nreturns K-pop songs"]
    end

    USER -->|"fills form"| FORM
    FORM -->|"user_prefs dict"| VECTORIZE
    VECTORIZE -->|"6-dim query vector"| COSINE
    KAGGLE -->|"build once ~5 s\ncached to disk"| MATRIX
    KAGGLE -->|"song rows"| INFER
    INFER -->|"mood-labeled song dicts"| MATRIX
    MATRIX -->|"load cached ~0.5 s"| COSINE
    COSINE -->|"50 candidate song dicts"| RERANK
    RERANK -->|"top-5 + scores + reasons"| RESULTS
    RESULTS -->|"displays"| USER
    RAG -->|"pytest -v"| UNIT
    RESULTS -->|"human checks quality"| HUMAN
```

