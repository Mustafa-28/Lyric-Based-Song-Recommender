!pip install gdown
!gdown https://drive.google.com/uc?id=151-heow7OrJqdNPfwplMFaldcgXoEeQ1
# IMPORTANT: Colab requires specific versions to avoid conflicts
# !pip uninstall -y gensim numpy
!pip install numpy==1.26.4 gensim --no-cache-dir

import gensim
from gensim.models import Word2Vec
import numpy
print("Gensim version:", gensim.__version__)
print("Numpy version:", numpy.__version__)

import nltk
nltk.download('punkt_tab')

import pandas as pd
import numpy as np
import string
import re
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('songdata.csv')
df.head()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.strip()]
    return tokens

df['text_clean'] = df['text'].apply(preprocess_text)
df.head()

corpus = df['text_clean'].tolist()

w2v_model1 = Word2Vec(
    sentences=corpus,
    vector_size=100,     # dimensionality of embeddings
    window=5,            # context window
    min_count=5,         # ignore words with total freq < 5
    workers=4,           # parallel training
    sg=1,                # 1 for skip-gram; 0 for CBOW
    negative=5,          # negative sampling
    sample=1e-5          # downsample frequent words
)

word_vectors = w2v_model1.wv

def get_song_vector(tokens, model):
    vectors = [model[word] for word in tokens if word in model]
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)  # mean across words in the song

df['song_vector'] = df['text_clean'].apply(lambda x: get_song_vector(x, word_vectors))

# STEP 4: Function to fetch song index & display lyrics
def get_song_index(song_name, artist_name=None):
    matches = df[df['song'].str.lower() == song_name.lower()]
    if artist_name:
        matches = matches[matches['artist'].str.lower() == artist_name.lower()]
    if matches.empty:
        raise ValueError("Song not found.")
    for i, row in matches.iterrows():
        print(f"\n🔹 Song: {row['song']} by {row['artist']}")
        print(f"🎵 Lyrics:\n{row['text']}")
        print("="*60)
    return matches.index[0]  # return first match

# STEP 5: Find similar songs
def find_similar_songs(song_name, artist_name=None, top_n=10):
    song_idx = get_song_index(song_name, artist_name)
    query_vec = df.loc[song_idx, 'song_vector'].reshape(1, -1)
    all_vectors = np.stack(df['song_vector'].values)
    similarities = cosine_similarity(query_vec, all_vectors)[0]
    top_indices = similarities.argsort()[-(top_n + 1):][::-1][1:]  # exclude query itself

    similar_songs = df.iloc[top_indices]

    for i, row in similar_songs.iterrows():
        print(f"\n🔹 Song: {row['song']} by {row['artist']}")
        print(f"🎵 Lyrics:\n{row['text']}")
        print("="*60)

    return similar_songs[['song', 'artist', 'text']]
  
find_similar_songs("Imagine", "The Beatles")
