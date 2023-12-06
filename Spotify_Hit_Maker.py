import streamlit as st

st.set_page_config(
    page_title="Spotify Top 2023",
    page_icon="📊",
)

st.sidebar.success("Pick a Tool to start 👆")
st.sidebar.write("put together by :blue[**Huy Pham**] 😿")
# Project Overview
st.markdown("## Hello 👋, Welcome to your next biggest :green[Spotify] hit 😲")
with st.expander("## Project Overview 🎵", expanded=True):

    st.markdown(
        """
        The **Spotify Hit-Maker** is an interactive tool designed to help you explore and understand the audio features 
        that make songs successful in 2023. Whether you're a music enthusiast, artist, or data science enthusiast, 
        this tool provides insights into the key elements that contribute to a song's popularity.

        ### Explore Audio Features 🎛️
        
        Dive into the terminology of audio features such as danceability, instrumentalness, liveness, valence, tempo, and more.
        Understand how these features contribute to the overall appeal of a song.

        ### Song Recommender 🤖

        Select a hit song from the Spotify top songs in 2023, and our Recommender will analyze its audio features to find 
        similar tracks. Discover new songs that share common characteristics with your favorite hits.

        ### Stream Predictor 🧙‍♂️

        Our Spotify stream Prophet 🔮 will help you predict the most likely stream count of your gonna-be-a-hit song. 
        Optimize your track's chance of success with our omnipotent predictor.   
        """
    )
    # Dataset information
    url = "https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023/data"

    ""
    ""
    "*Kaggle Dataset:*"
    st.markdown(f"[Spotify top songs in 2023 dataset]({url})")

