# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  
Example: **VibeFinder**  

---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 

Prompts:  

- What kind of recommendations does it generate  

Given a user's taste profile (genre, mood, energy level, positivity, and acoustic preference), the system scores every song in the catalog and returns the top 3 best-matching songs, along with a plain-English explanation of why each song was suggested

- What assumptions does it make about the user  

The system assumes a user has a single fixed "taste profile" — one favourite genre, one favourite mood, and specific numerical preferences. 

- Is this for real users or classroom exploration  

 This is a classroom/educational project — it is a simulation designed to help students understand how scoring-based recommenders work, not a deployed product

---

## 3. How the Model Works  

Explain your scoring approach in simple language.  

Prompts:  

- What features of each song are used (genre, energy, mood, etc.)  

genre, mood, energy (how active it feels), valence (how positive/happy it sounds), and acousticness (how acoustic vs. electronic it is). 

- What user preferences are considered  

 favourite genre, favourite mood, desired energy level, desired positivity (valence), and preferred acousticness

- How does the model turn those into a score  

- Genre and mood are checked as a yes/no match — a perfect match gives full points, a mismatch gives zero
- Energy, valence, and acousticness are scored using a "closeness curve" — the closer the song's value is to what the user wants, the higher the score, and it fades off gradually (never completely zero)
- All five scores are weighted and averaged into a final score between 0 and 1
- To add variety, the results also enforce that no more than 2 songs from the same genre appear and near-duplicate songs are skipped

- What changes did you make from the starter logic  

I change the weight fromt he starter logic due to mismatch output

Avoid code here. Pretend you are explaining the idea to a friend who does not program.

---

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  

There are 18 songs in the dataset

- What genres or moods are represented  

Genres (10): pop, lofi, rock, ambient, jazz, synthwave, indie pop, country, hip-hop, classical, r&b, metal, folk, indie folk, edm
Moods (9): happy, chill, intense, relaxed, moody, focused, nostalgic, energetic, sad, romantic
Numerical features per song: energy (0–1), tempo (BPM), valence (positivity), danceability, acousticness

- Did you add or remove data  

Yes , I added 8 external songs for diversity 

- Are there parts of musical taste missing in the dataset  

1. Too few songs — 18 songs is very small; a real recommender needs hundreds or thousands for meaningful patterns
2. No scores/popularity — missing play count, rating, or listener data that often drives recommendations
3. Missing subgenres — no reggae, blues, latin, K-pop, gospel, trap, soul, or world music
4. No explicit tempo variety within genres — e.g., only one or two songs per genre, so the model can't distinguish preferences within a genre
5. No negative moods beyond "sad" — e.g., angry, melancholic, anxious are absent
6. No featured artist / collaboration data — limits artist-based similarity
7. Single representation per mood — moods like "focused" and "nostalgic" appear only once, making them statistically underrepresented
8. No release year — can't distinguish between retro vs. modern taste preferences

---

## 5. Strengths  

Where does your system seem to work well  

Prompts:  

- User types for which it gives reasonable results  
- Any patterns you think your scoring captures correctly  
- Cases where the recommendations matched your intuition  

1. each recommendation has given comparison matric with the expected output and model has good matches with pop and happy genre songs 

---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

Prompts:  

- Features it does not consider  
- Genres or moods that are underrepresented  
- Cases where the system overfits to one preference  
- Ways the scoring might unintentionally favor some users  

Some dataset like metal, classical, ambient, edm only appear once in the songs.csv file. Hence , for users who like this king of songs will not get a lot of option and more choice of song. 

The current music recommendtion model only support single point prefrence, which mean user with multiple prefrence ( like they prefre both pop and chill) are forced to pick one side, which lacks of diversity 

---

## 7. Evaluation  

How you checked whether the recommender behaved as expected. 

Prompts:  

- Which user profiles you tested  
- What you looked for in the recommendations  
- What surprised you  
- Any simple tests or comparisons you ran  

No need for numeric metrics unless you created some.
Based on my result when running main.py , the model performed as expected and generates output that matches with the expected output. One things I notoced is that the song 'Gym Hero' fits into multiple category.

1. Generated test case in test_recommender.py
2. evaluatethe result by running main.py 


---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
- Better ways to explain recommendations  
- Improving diversity among the top results  
- Handling more complex user tastes  

---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
- Something unexpected or interesting you discovered  
- How this changed the way you think about music recommendation apps  

Throughout this project , I gained deep understanding of potential edge cases that a recommendation system will works. In real world cases , people will have multiple prefrence in music and the system has to provide the desired output based on the prefrence. Given that there s still a lot of space of improvement for this project , I would be highly interested to continue on this as my project , to refine the model that fist a real world scenario
