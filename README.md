# Lyric-Based-Song-Recommender
A Python NLP project that finds semantically similar songs by analyzing lyrical content using Word2Vec embeddings.

Key Features
Recommends top-N similar songs based on lyrical meaning

Processes raw lyrics (punctuation removal, tokenization, lowercase)

Generates song embeddings by averaging Word2Vec vectors

Uses cosine similarity to measure lyrical similarity

Simple CLI interface (input song + artist → get recommendations)

Technical Details
Model: Word2Vec (Gensim/PyTorch implementation)

Dataset: Custom song-lyrics corpus (e.g., songdata.csv)

NLP Tools: NLTK, regex, pandas

Similarity Metric: Cosine distance

Usage
Install dependencies:

bash
pip install gensim==4.3.1 numpy==1.26.4 nltk pandas sklearn
Run the script:

python
from recommender import find_similar_songs
find_similar_songs("Imagine", "The Beatles", top_n=5)  # Example
Example Output
🔹 Song: Let It Be by The Beatles  
🎵 Lyrics: "When I find myself in times of trouble..."  
==================================================  
🔹 Song: Hey Jude by The Beatles  
🎵 Lyrics: "Hey Jude, don’t make it bad..."  
Project Structure
├── data/                  # Lyrics dataset (e.g., songdata.csv)  
├── model/                 # Saved Word2Vec embeddings  
├── preprocessing.py       # Text cleaning utils  
├── recommender.py         # Core similarity logic  
└── demo.ipynb             # Jupyter notebook example  
Applications
Music discovery

Playlist generation

NLP/embedding tutorials

Note: Replace songdata.csv with your dataset. Model can be retrained for other genres.

