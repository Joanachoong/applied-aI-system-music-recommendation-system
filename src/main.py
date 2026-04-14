"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv") 

    # "Upbeat pop fan" taste profile
    # Represents someone who loves bright, produced pop with high energy and positive vibes.
    # Keys match UserProfile field names exactly.
    user_prefs = {
        "favorite_genre":         "pop",
        "favorite_mood":          "happy",
        "target_energy":          0.78,   # active but not maxed-out
        "preferred_valence":      0.80,   # strongly positive/happy tone
        "preferred_acousticness": 0.22,   # produced sound with just a little warmth
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nTop recommendations:\n")
    for rec in recommendations:
        # You decide the structure of each returned item.
        # A common pattern is: (song, score, explanation)
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"Because: {explanation}")
        print()


if __name__ == "__main__":
    main()
