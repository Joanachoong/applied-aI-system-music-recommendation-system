# 🎵 Music Recommender Simulation

## Project Summary

In this project you will build and explain a small music recommender system.

Your goal is to:

- Represent songs and a user "taste profile" as data
- Design a scoring rule that turns that data into recommendations
- Evaluate what your system gets right and wrong
- Reflect on how this mirrors real world AI recommenders

Replace this paragraph with your own summary of what your version does.

---

## How The System Works

Explain your design in plain language.

Some prompts to answer:

- What features does each `Song` use in your system
  - For example: genre, mood, energy, tempo
- What information does your `UserProfile` store
- How does your `Recommender` compute a score for each song
- How do you choose which songs to recommend

You can include a simple diagram or bullet list if helpful.

My main approach of designing this music recoomendation system is based on 2 rules

1. Scoring rule 
2. Ranking rule

Scoring Rule:   score(song, user_prefs) → float between 0 and 1
                (Gaussian similarity + categorical match weights)

Ranking Rule:   rank(scored_songs, limit=5) → top N songs
                Rules:
                  1. Sort by score descending (primary)
                  2. No more than 2 songs of the same genre in results
                  3. If scores within 0.05 of each other, prefer different mood
                  4. Return top N results

### Why I set 2 rules 
Scoring rules show the result of songs from 0-1 , while Ranking rules will dsplay the song in descending order , meaning score with highest ranking will probably fits the user taste the most

### Simple math based 'Scoring Rule'

total_score = (w_e × score_energy
             + w_v × score_valence
             + w_a × score_acousticness
             + w_mood × mood_match
             + w_genre × genre_match)
             ─────────────────────────────
               (w_e + w_v + w_a + w_mood + w_genre)

| Feature | Weight | Reasoning |
|---|---|---|
| `valence` | 0.30 | Strongest mood anchor — happy vs sad |
| `mood` | 0.25 | User's explicit intent |
| `energy` | 0.20 | Activity level driver |
| `genre` | 0.15 | Style preference, partially redundant with numericals |
| `acousticness` | 0.10 | Texture detail, lowest unique signal |
| **Total** | **1.00** | |



- this formula will score songs based of user prefrence ( highest is 1.0 , lowest is 0)
category_score = 1.0  if song value == user preference
               = 0.0  if no match

### Song Features Used in Simulation

| Field | Type | Role |
|---|---|---|
| `id` | `int` | Unique identifier |
| `title` | `str` | Display name |
| `artist` | `str` | Display name |
| `genre` | `str` | Categorical scoring (weight 0.15) |
| `mood` | `str` | Categorical scoring (weight 0.25) |
| `energy` | `float` | Gaussian scoring (weight 0.20) |
| `valence` | `float` | Gaussian scoring (weight 0.30) |
| `acousticness` | `float` | Gaussian scoring (weight 0.10) |
| `tempo_bpm` | `float` | Stored, not scored |
| `danceability` | `float` | Stored, not scored |

### UserProfile Features Used in Simulation

| Field | Type | Matches Against |
|---|---|---|
| `favorite_genre` | `str` | `song.genre` |
| `favorite_mood` | `str` | `song.mood` |
| `preferred_energy` | `float` | `song.energy` |
| `preferred_valence` | `float` | `song.valence` |
| `preferred_acousticness` | `float` | `song.acousticness` |

---

## Getting Started

### Setup

1. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Mac or Linux
   .venv\Scripts\activate         # Windows

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
python -m src.main
```

### Running Tests

Run the starter tests with:

```bash
pytest
```

You can add more tests in `tests/test_recommender.py`.

---

## Experiments You Tried

Use this section to document the experiments you ran. For example:

- What happened when you changed the weight on genre from 2.0 to 0.5
- What happened when you added tempo or valence to the score
- How did your system behave for different types of users

---

## Limitations and Risks

Summarize some limitations of your recommender.

Examples:

- It only works on a tiny catalog
- It does not understand lyrics or language
- It might over favor one genre or mood

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Write 1 to 2 paragraphs here about what you learned:

- about how recommenders turn data into predictions
- about where bias or unfairness could show up in systems like this


---

## 7. `model_card_template.md`

Combines reflection and model card framing from the Module 3 guidance. :contentReference[oaicite:2]{index=2}  

```markdown
# 🎧 Model Card - Music Recommender Simulation

## 1. Model Name

Give your recommender a name, for example:

> VibeFinder 1.0

---

## 2. Intended Use

- What is this system trying to do
- Who is it for

Example:

> This model suggests 3 to 5 songs from a small catalog based on a user's preferred genre, mood, and energy level. It is for classroom exploration only, not for real users.

---

## 3. How It Works (Short Explanation)

Describe your scoring logic in plain language.

- What features of each song does it consider
- What information about the user does it use
- How does it turn those into a number

Try to avoid code in this section, treat it like an explanation to a non programmer.

---

## 4. Data

Describe your dataset.

- How many songs are in `data/songs.csv`
- Did you add or remove any songs
- What kinds of genres or moods are represented
- Whose taste does this data mostly reflect

---

## 5. Strengths

Where does your recommender work well

You can think about:
- Situations where the top results "felt right"
- Particular user profiles it served well
- Simplicity or transparency benefits

---

## 6. Limitations and Bias

Where does your recommender struggle

Some prompts:
- Does it ignore some genres or moods
- Does it treat all users as if they have the same taste shape
- Is it biased toward high energy or one genre by default
- How could this be unfair if used in a real product

---

## 7. Evaluation

How did you check your system

Examples:
- You tried multiple user profiles and wrote down whether the results matched your expectations
- You compared your simulation to what a real app like Spotify or YouTube tends to recommend
- You wrote tests for your scoring logic

You do not need a numeric metric, but if you used one, explain what it measures.

---

## 8. Future Work

If you had more time, how would you improve this recommender

Examples:

- Add support for multiple users and "group vibe" recommendations
- Balance diversity of songs instead of always picking the closest match
- Use more features, like tempo ranges or lyric themes

---

## 9. Personal Reflection

A few sentences about what you learned:

- What surprised you about how your system behaved
- How did building this change how you think about real music recommenders
- Where do you think human judgment still matters, even if the model seems "smart"

