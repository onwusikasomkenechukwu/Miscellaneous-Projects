import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

import os


st.write("Current working directory:", os.getcwd())
st.write("Files in current directory:", os.listdir())


if not os.path.exists("top_10000_1950-now.csv"):
    st.error("Dataset not found. Please upload or check the file path.")
    st.stop()

# Load the data
df = pd.read_csv("top_10000_1950-now.csv")
df = df.dropna(subset=['Track Name', 'Artist Name(s)', 'Track Duration (ms)',
                       'Danceability', 'Energy', 'Tempo', 'Album Image URL', 'Track URI'])

# Clean column for easier matching
df['Track Name Lower'] = df['Track Name'].str.lower()

# Standardize features
features = ['Danceability', 'Energy', 'Tempo']
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Compute similarity matrix
similarity_matrix = cosine_similarity(df_scaled)

# Recommendation function
def recommend_next_song(song_index, used_indices, top_n=10):
    similarities = similarity_matrix[song_index]
    similar_indices = np.argsort(similarities)[::-1]
    for idx in similar_indices:
        if idx != song_index and idx not in used_indices:
            return idx
    return None

# Streamlit UI
st.title("🎵 Spotify Song Recommender")

user_input = st.text_input("Enter the name of a song:", "Shape of You")

if user_input:
    matches = df[df['Track Name Lower'] == user_input.lower()]
    if matches.empty:
        st.error("Song not found. Try another one.")
    else:
        start_index = matches.index[0]
        playlist_indices = [start_index]

        st.subheader("Now Playing:")
        song = df.loc[start_index]
        track_uri = song['Track URI'].split(":")[-1]
        duration_ms = song['Track Duration (ms)']
        minutes = int(duration_ms // 60000)
        seconds = int((duration_ms % 60000) // 1000)
        duration_str = f"{minutes}:{seconds:02d}"

        st.image(song['Album Image URL'], width=150)
        st.markdown(f"**{song['Track Name']}** by *{song['Artist Name(s)']}*  ")
        st.markdown(f"Duration: {duration_str}  ")
        st.markdown(f"[Listen on Spotify](https://open.spotify.com/track/{track_uri})")

        # Generate recommendations
        st.subheader("Up Next:")
        for _ in range(5):
            next_index = recommend_next_song(playlist_indices[-1], playlist_indices)
            if next_index is None:
                break
            playlist_indices.append(next_index)
            song = df.loc[next_index]
            track_uri = song['Track URI'].split(":")[-1]
            duration_ms = song['Track Duration (ms)']
            minutes = int(duration_ms // 60000)
            seconds = int((duration_ms % 60000) // 1000)
            duration_str = f"{minutes}:{seconds:02d}"

            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(song['Album Image URL'], width=80)
            with col2:
                st.markdown(f"**{song['Track Name']}** by *{song['Artist Name(s)']}*")
                st.markdown(f"Duration: {duration_str}")
                st.markdown(f"[Listen on Spotify](https://open.spotify.com/track/{track_uri})")
