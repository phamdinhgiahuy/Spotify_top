import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
from PIL import Image
import altair as alt
import re
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Song Recommender", page_icon="🤖")
st.markdown("## Learn from their :orange[**success stories**] 🎉🥇🍾")
st.markdown(
    """
    To pave our ways for success. We need first to examine these :blue[similar tracks] to the :red[hit] you chose.
    What do they have in common? Any striking *"audio features"* we need to pay attention to in order to best replicate the world shattering 
    impact these song has on the musical scene in 2023?
"""
)

with st.expander("Peek at 🎛️:violet[audio features] terminology 🎼"):
    st.markdown(
        """
        :red[**Danceability:**]

        How suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity

        :red[**Instrumentalness:**]

        The likelihood of a track contains no vocals

        :red[**Liveness:**]

        Detects the presence of an audience in the recording.

        :red[**Valence:**]

        Describing the musical positiveness conveyed by a track

        :red[**Tempo:**]

        The overall estimated tempo of a track in beats per minute (BPM). 
        In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.       

        :red[**Instrumentalness:**]

        The likelihood of a track contains no vocals

        :red[**Key:**]

        The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.

        :red[**Energy:**]

        Represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy.

    """)   
    #st.image("https://static.streamlit.io/examples/dice.jpg")

#Load dataset
spotify = pd.read_csv(r"C:\Users\Admin\OneDrive - Michigan State University\Courses\CMSE 830\Project\Dataset\spotify-2023.csv", encoding='ISO-8859-1')

#Pre processing
spotify_ml = spotify.copy()
spotify_ml['released_date'] = pd.to_datetime(spotify_ml['released_year'].astype(str) + '-' +
                        spotify_ml['released_month'].astype(str) + '-' +
                        spotify_ml['released_day'].astype(str), errors='coerce')
spotify_ml['released_date'] = spotify_ml['released_date'].apply(lambda x: x.timestamp())
drop_col = ['in_apple_playlists', 'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts', 'in_shazam_charts', 'released_year', 'released_month', 'released_day']
spotify_ml = spotify_ml.drop(drop_col, axis=1)
spotify_ml.streams = spotify_ml.streams.apply(lambda x: pd.to_numeric(x ,errors='coerce'))
spotify_ml['key'] = spotify_ml['key'].fillna(-1)
spotify_ml.dropna(subset=['streams'], inplace=True)
mode_code = pd.get_dummies(spotify_ml['mode'], prefix='mode')
spotify_ml = pd.concat([spotify_ml, mode_code], axis=1)
encoder = LabelEncoder()
spotify_ml['key_encoded'] = encoder.fit_transform(spotify_ml['key'].astype(str))
percent_col = [i for i in spotify.columns if i.find('_%') != -1]
for col in percent_col:
    spotify_ml[col.replace('_%', '')] = spotify_ml[col] / 100.0

ml_dropcol = [
    'danceability_%', 'valence_%', 'energy_%', 
    'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%', 'key', 'mode'                      
]
spotify_ml = spotify_ml.drop(ml_dropcol, axis=1)

track_sel = st.sidebar.selectbox("Which hit should we replicate?", spotify_ml['track_name'], index=None, placeholder="Type a song ...")
rec_num = st.sidebar.number_input("I need to examine this many similar hits:", min_value=0, max_value=15, step=1, value=10)
if track_sel:
    st.write('The Recommender 🥁🤖 found', rec_num, 'similar tracks to the hit:    ', track_sel)
    #Recommendation DF
    rec_col = ['track_name', 'artist(s)_name', 'bpm', 'danceability', 'valence', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
    spotify_rec = spotify_ml[rec_col]
    scaler = MinMaxScaler()
    numeric_columns = list(spotify_rec.select_dtypes(include=[np.number]).columns)
    spotify_rec[numeric_columns] = scaler.fit_transform(spotify_rec[numeric_columns])


    def recommend(df_, track, amount=5):
        song = df_[df_['track_name'].str.lower() == track.lower()].head(1)
        rec = df_[df_['track_name'].str.lower() != track.lower()]

        # Calculate cosine similarity
        similarity_scores = cosine_similarity(song[numeric_columns], rec[numeric_columns])[0]

        # Add similarity scores to the DataFrame
        rec['similarity'] = similarity_scores

        # Sort by similarity and select top recommendations
        rec = rec.sort_values('similarity', ascending=False)
        
        return rec[:amount]

    recommendations = recommend(spotify_rec, track_sel, rec_num).reset_index()

    recommendations[['track_name', 'artist(s)_name']]
        
    st.markdown(
        """
        ### Visualize Recommendations 📊
        Visualize the recommended songs based on their musicality and other primary features. The tool provides charts and 
        scatter plots to help you explore the relationships between different attributes.
        """)
    rec_li = list(recommendations['track_name'])
    rec_li.append(track_sel)
    atrr_col = spotify_ml.columns[11:18]
    key_col = [i for i in spotify_ml.columns if i not in atrr_col]
    mask = spotify_ml['track_name'].isin(rec_li)
    rec_df = spotify_ml[mask]
    rec_df_melt = rec_df.melt(id_vars=key_col, value_vars=atrr_col,
                            var_name='attr', value_name='values')
    top_n = (rec_num+1)*3  

    # Filter data to include only the top N values
    top_n_data = rec_df_melt.sort_values('values', ascending=False).head(top_n)

    # Bar Plot
    bar_plot = alt.Chart(rec_df_melt).mark_bar().encode(
        y='track_name:N',
        x=alt.X('sum(values):Q').stack('zero'),
        color='attr',
        tooltip=['track_name:N', 'artist(s)_name:N', 'streams:Q'],
        order=alt.Order(
        'values',
        sort='descending'
        )
    ).properties(
        title='Recommended Songs Musicality',
        width=500,
        height=200
    )

    # Text annotations for top N values
    text = alt.Chart(top_n_data).mark_text(dx=-15, dy=1, color='black').encode(
        y='track_name:N',
        x=alt.X('sum(values):Q').stack('zero'),
        detail='attr',
        text=alt.Text('sum(values):Q', format='.2f'),
        order=alt.Order(
        'values',
        sort='descending'
        )
    ).properties(
        width=500,
        height=200
    )

    # Scatter Plot
    scatter_plot = alt.Chart(rec_df_melt).mark_circle().encode(
        x='track_name:N',
        y='bpm:Q',
        size='streams:Q',
        color='key_encoded:N',
        tooltip=['track_name:N', 'artist(s)_name:N', 'bpm:Q', 'streams:Q', 'artist_count']
    ).properties(
        title='Recommended Songs Primary Features',
        width=400,
        height=100
    )

    # Combine plots
    rec_vis = (bar_plot + text) & scatter_plot

    st.altair_chart(rec_vis, use_container_width=True)

