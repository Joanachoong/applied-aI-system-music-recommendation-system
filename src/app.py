"""
Streamlit UI for the RAG-powered music recommender.

Run with:
    streamlit run src/app.py
"""

import sys
import os

# Ensure the project root is on sys.path so 'src.*' imports resolve
# regardless of how Streamlit launches the script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.rag import get_or_build_index, rag_recommend

# ── Constants ──────────────────────────────────────────────────────────────────

KAGGLE_CSV = "data/spotify_tracks.csv"
INDEX_PATH = "data/rag_index.npz"

ALL_GENRES = [
    "acoustic", "afrobeat", "alt-rock", "alternative", "ambient", "anime",
    "black-metal", "bluegrass", "blues", "brazil", "breakbeat", "british",
    "cantopop", "chicago-house", "children", "chill", "classical", "club",
    "comedy", "country", "dance", "dancehall", "death-metal", "deep-house",
    "detroit-techno", "disco", "disney", "drum-and-bass", "dub", "dubstep",
    "edm", "electro", "electronic", "emo", "folk", "forro", "french", "funk",
    "garage", "german", "gospel", "goth", "grindcore", "groove", "grunge",
    "guitar", "happy", "hard-rock", "hardcore", "hardstyle", "heavy-metal",
    "hip-hop", "honky-tonk", "house", "idm", "indian", "indie", "indie-pop",
    "industrial", "iranian", "j-dance", "j-idol", "j-pop", "j-rock", "jazz",
    "k-pop", "kids", "latin", "latino", "malay", "mandopop", "metal",
    "metalcore", "minimal-techno", "mpb", "new-age", "opera", "pagode",
    "party", "piano", "pop", "pop-film", "power-pop", "progressive-house",
    "psych-rock", "punk", "punk-rock", "r-n-b", "reggae", "reggaeton", "rock",
    "rock-n-roll", "rockabilly", "romance", "sad", "salsa", "samba",
    "sertanejo", "show-tunes", "singer-songwriter", "ska", "sleep",
    "songwriter", "soul", "spanish", "study", "swedish", "synth-pop", "tango",
    "techno", "trance", "trip-hop", "turkish", "world-music",
]

ALL_MOODS = ["happy", "sad", "chill", "relaxed", "intense",
             "energetic", "focused", "romantic", "nostalgic", "moody"]


# ── Cached index loader (built once per Streamlit session) ─────────────────────

@st.cache_resource(show_spinner="Loading 114k song index… (first run ~10 s)")
def load_index_cached():
    return get_or_build_index(KAGGLE_CSV, INDEX_PATH)


# ── UI helpers ─────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render the preference form in the sidebar and return a user_prefs dict."""
    st.sidebar.title("🎵 Music Recommender")
    st.sidebar.markdown("Tell us what you're in the mood for:")

    genre = st.sidebar.selectbox(
        "What genre do you like?",
        options=ALL_GENRES,
        index=ALL_GENRES.index("pop"),
    )

    mood = st.sidebar.selectbox(
        "What mood are you in?",
        options=ALL_MOODS,
        index=ALL_MOODS.index("happy"),
    )

    energy = st.sidebar.slider(
        "Energy level  (low ← → high)",
        min_value=0.0, max_value=1.0, value=0.70, step=0.05,
    )

    valence = st.sidebar.slider(
        "Positivity / Valence  (melancholic ← → uplifting)",
        min_value=0.0, max_value=1.0, value=0.60, step=0.05,
    )

    acousticness = st.sidebar.slider(
        "Acoustic feel  (electronic ← → acoustic)",
        min_value=0.0, max_value=1.0, value=0.30, step=0.05,
    )

    st.sidebar.markdown("---")
    go = st.sidebar.button("🎶 Get Recommendations", use_container_width=True)

    return {
        "favorite_genre": genre,
        "favorite_mood": mood,
        "target_energy": energy,
        "preferred_valence": valence,
        "preferred_acousticness": acousticness,
    }, go


def render_results(recommendations: list) -> None:
    """Display top-5 song cards with score bars and reason breakdowns."""
    st.subheader("🎧 Your Top 5 Picks")

    if not recommendations:
        st.warning("No songs found. Try adjusting your preferences.")
        return

    for rank, (song, score, reasons) in enumerate(recommendations, start=1):
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f"**{rank}. {song['title']}** &nbsp; — &nbsp; *{song['artist']}*"
                )
                st.caption(
                    f"genre: `{song['genre']}`  •  mood: `{song['mood']}`  •  "
                    f"energy: `{song['energy']:.2f}`  •  valence: `{song['valence']:.2f}`"
                )
            with col2:
                st.metric(label="Match score", value=f"{score:.2f}")

            st.progress(score, text="")

            with st.expander("Why this song?"):
                for reason in reasons:
                    st.markdown(f"- {reason}")

            st.divider()


def render_explainer() -> None:
    """Show an educational explanation of the RAG pipeline."""
    with st.expander("ℹ️ How the RAG system works"):
        st.markdown(
            """
**Retrieval-Augmented Generation (RAG)** extends the recommender from 18 songs → 114,000 real Spotify tracks.

**Step-by-step:**

1. **You fill the form** — genre, mood, energy, valence, acousticness become a *user profile*.
2. **Profile → 6-dim vector** — your preferences are converted into a numeric vector of audio features.
3. **Cosine similarity search** — that vector is compared against all 114,000 song vectors in the index.
   Songs that *sound* like your preferences float to the top (no genre filter yet).
4. **Top-50 candidates retrieved** — the 50 most sonically similar songs are selected.
5. **Scoring engine re-ranks** — the existing weighted scorer (genre +0.25, mood +0.25, valence +0.25,
   energy +0.20, acousticness +0.05) picks the best 5 from those 50 candidates.
6. **You see the results** — top-5 songs with match scores and a reason for each feature.

> **Why RAG?** Without retrieval, searching for "k-pop" returns nothing because the original dataset
> has no k-pop tracks. With RAG, we pull real k-pop songs from Spotify before scoring.
            """
        )


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Music Recommender",
        page_icon="🎵",
        layout="wide",
    )

    st.title("🎵 Music Recommender — RAG Edition")
    st.markdown(
        "Powered by **114,000 Spotify tracks** and audio-feature vector similarity. "
        "Fill in your preferences on the left and hit **Get Recommendations**."
    )

    user_prefs, go = render_sidebar()

    render_explainer()

    if go:
        song_dicts, matrix_normed = load_index_cached()
        with st.spinner("Finding your songs…"):
            recommendations = rag_recommend(
                user_prefs, song_dicts, matrix_normed,
                k_retrieve=50, k_final=5,
            )
        render_results(recommendations)
    else:
        st.info("Select your preferences in the sidebar and click **🎶 Get Recommendations**.")


if __name__ == "__main__":
    main()
