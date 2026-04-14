"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from src.recommender import load_songs, recommend_songs


def _print_profile_report(profile_name: str, user_prefs: dict, predicted_top_titles: list, songs: list) -> None:
    """Print expected vs actual top recommendations for one profile."""
    recommendations = recommend_songs(user_prefs, songs, k=3)
    actual_top_titles = [song["title"] for song, _, _ in recommendations]

    print(f"\n=== {profile_name} ===")
    print("Your predicted top songs:")
    if predicted_top_titles:
        for idx, title in enumerate(predicted_top_titles, start=1):
            print(f"  {idx}. {title}")
    else:
        print("  (No predictions entered yet)")

    print("Model top 3 songs:")
    for idx, rec in enumerate(recommendations, start=1):
        song, score, reasons = rec
        print(f"  {idx}. {song['title']} - {score:.2f}")
        for reason in reasons:
            print(f"     - {reason}")

    if predicted_top_titles:
        overlap = [title for title in actual_top_titles if title in predicted_top_titles]
        print(f"Overlap count: {len(overlap)} / {len(predicted_top_titles)}")
        if overlap:
            print("Matching songs:")
            for title in overlap:
                print(f"  - {title}")
        else:
            print("Matching songs: none")


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    evaluation_profiles = [
        {
            "name": "High-Energy Pop",
            "user_prefs": {
                "favorite_genre": "pop",
                "favorite_mood": "happy",
                "target_energy": 0.90,
                "preferred_valence": 0.85,
                "preferred_acousticness": 0.15,
            },
            "predicted_top_titles": [
                "Sunrise City",
                "Gym Hero",
                "Rooftop Lights",
            ],
        },
        {
            "name": "Chill Lofi",
            "user_prefs": {
                "favorite_genre": "lofi",
                "favorite_mood": "calm",
                "target_energy": 0.25,
                "preferred_valence": 0.40,
                "preferred_acousticness": 0.85,
            },
            "predicted_top_titles": [
                "Library Rain",
                "Midnight Coding",
                "Focus Flow",
            ],
        },
        {
            "name": "Deep Intense Rock",
            "user_prefs": {
                "favorite_genre": "rock",
                "favorite_mood": "intense",
                "target_energy": 0.95,
                "preferred_valence": 0.30,
                "preferred_acousticness": 0.10,
            },
            "predicted_top_titles": [
                "Storm Runner",
                "Iron Cathedral",
                "Gym Hero",
            ],
        },
        {
            "name": "Edge Case: Conflicting Signals",
            "user_prefs": {
                "favorite_genre": "pop",
                "favorite_mood": "sad",
                "target_energy": 0.90,
                "preferred_valence": 0.10,
                "preferred_acousticness": 0.90,
            },
            "predicted_top_titles": [
                "Quiet Ember",
                "Last Train Home",
                "Gym Hero",
            ],
        },
    ]

    print("\nSystem evaluation output:")
    print("Edit predicted_top_titles in src/main.py to compare your own expectations.")

    for profile in evaluation_profiles:
        _print_profile_report(
            profile_name=profile["name"],
            user_prefs=profile["user_prefs"],
            predicted_top_titles=profile["predicted_top_titles"],
            songs=songs,
        )


if __name__ == "__main__":
    main()
