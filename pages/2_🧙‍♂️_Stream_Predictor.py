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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import time

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay, f1_score

st.set_page_config(page_title="Stream Predictor", page_icon="🧙‍♂️")

# User Input for Song Attributes
st.sidebar.header("Input Song Attributes")
in_spotify_playlists = st.sidebar.number_input("Make it to # Spotify Playlists Goal:", min_value=0, max_value=145, step=1, value=20)
bpm = st.sidebar.slider("BPM", 60, 210, 80, 1)
danceability = st.sidebar.slider("Danceability", 0.0, 1.0, 0.5, 0.1)
valence = st.sidebar.slider("Valence", 0.0, 1.0, 0.5, 0.1)
energy = st.sidebar.slider("Energy", 0.0, 1.0, 0.5, 0.1)
acousticness = st.sidebar.slider("Acousticness", 0.0, 1.0, 0.5, 0.1)
instrumentalness = st.sidebar.slider("Instrumentalness", 0.0, 1.0, 0.5, 0.1)
liveness = st.sidebar.slider("Liveness", 0.0, 1.0, 0.5, 0.1)
speechiness = st.sidebar.slider("Speechiness", 0.0, 1.0, 0.5, 0.1)
key_encoded = st.sidebar.selectbox("Key", list(range(12)), 0)

# Create a DataFrame for user input
user_input = pd.DataFrame({
    'in_spotify_playlists': [in_spotify_playlists],
    'bpm' : [bpm],    
    'key_encoded': [key_encoded],    
    'danceability': [danceability],
    'valence': [valence],
    'energy': [energy],
    'acousticness': [acousticness],
    'instrumentalness': [instrumentalness],
    'liveness': [liveness],
    'speechiness': [speechiness]
})

if st.sidebar.button("Predict 🌠"):
    with st.status("Building the Stream Predictor...", expanded=True) as status:
        st.write("Preparing the dataset...")
            #Load dataset
        spotify = pd.read_csv(r"spotify-2023.csv", encoding='ISO-8859-1')

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

        numeric_data = spotify_ml.select_dtypes(include=[np.number])
        # Split the data into train-validation-test sets
        num_col = list(numeric_data.columns)
        ignore_col = ['streams', 'artist_count', 'released_date', 'in_spotify_charts']
        cols_to_use = [i for i in num_col if i not in ignore_col]
        X = spotify_ml[cols_to_use]
        y = spotify_ml['streams']

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=0)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)
        time.sleep(0.5)
        st.write("Data is nice and clean, ready to be baked 🍰🥝")
        time.sleep(0.5)
        st.write("Training the data ⚒️...")

        # Define the hyperparameter ranges
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_features': ['sqrt'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }


        # Initialize RandomForestRegressor with GridSearchCV or GradientBoostingRegressor()
        grid_search = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_grid, 
                                cv=3, n_jobs=-1, verbose=3, scoring='r2')

        # Fit the model
        grid_search.fit(X_train, y_train)

        # Print the best parameters
        print("Best parameters:", grid_search.best_params_)

        # Predict using the best model
        y_val_pred = grid_search.best_estimator_.predict(X_val)

        # Evaluate the model's performance
        mse = mean_squared_error(y_val, y_val_pred)
        mae = mean_absolute_error(y_val, y_val_pred)
        r2 = r2_score(y_val, y_val_pred)

        st.write("Testing the Predicter 🤺...")
        time.sleep(0.5)
        st.write(f"  Mean Squared Error (MSE): {mse}")
        time.sleep(0.5)
        st.write(f"  Mean Absolute Error (MAE): {mae}")
        time.sleep(0.5)
        st.write(f"  R^2 Score: {r2}")
        time.sleep(0.5)

        importances = grid_search.best_estimator_.feature_importances_
        features = cols_to_use


        # Create a DataFrame for the importances
        df_importances = pd.DataFrame({'Feature': features, 'Importance': importances})
        df_importances = df_importances.sort_values(by='Importance', ascending=True)
        time.sleep(0.5)
        status.update(label="Predicter is up and going!", state="complete", expanded=False)

    st.button('Retrain')



    # Create an interactive bar chart with Plotly Express using gradient color scale
    fig = px.bar(df_importances, x='Importance', y='Feature', orientation='h',
                title='Where investment should go',
                labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
                height=500,
                color='Importance',  # Assign importance scores as the color variable
                color_continuous_scale='tealgrn')  # Use the tealgrn color scale

    # Customize the layout
    fig.update_layout(xaxis=dict(title='Importance Score'),
                    yaxis=dict(title='Feature'),
                    coloraxis_colorbar=dict(title='Importance'),
                    showlegend=False)

    st.plotly_chart(fig, use_container_width=True)

    spotify_ml['Predicted_Streams'] = grid_search.best_estimator_.predict(X)

    # Create a DataFrame for visualization
    visualization_df = spotify_ml[['track_name', 'artist(s)_name', 'streams', 'Predicted_Streams']]

    # Scatter plot
    scatter_plot = alt.Chart(visualization_df).mark_circle(opacity=0.7, size=60).encode(
        x=alt.X('streams:Q', title='Actual Stream'),
        y=alt.Y('Predicted_Streams:Q', title='Predicted Stream'),
        tooltip=['track_name', 'artist(s)_name', 'streams', 'Predicted_Streams']
    ).properties(
        title='Actual vs Predicted Stream Counts',
        width=600,
        height=400
    )

    # Regression line
    regression_line = scatter_plot.transform_regression(
        'streams', 'Predicted_Streams'
    ).mark_line(color='red')

    # Residuals plot
    residuals_plot = alt.Chart(visualization_df).mark_bar().encode(
        x=alt.X('residuals:Q', title='Residuals'),
        y='count()',
        tooltip=['track_name', 'artist(s)_name']
    ).transform_calculate(
        residuals='datum.streams - datum.Predicted_Streams'
    ).properties(
        title='Residuals Plot',
        width=600,
        height=150
    )

    # Combine the plots
    final_chart = (scatter_plot + regression_line) & residuals_plot
    st.altair_chart(final_chart, use_container_width=True)


    # Predict using the best model
    #user_input_scaled = scaler.transform(user_input)
    predicted_streams = grid_search.best_estimator_.predict(user_input)

    # Display Prediction
    st.sidebar.header(":orange[Predicted Streams:]")
    st.sidebar.write(f"{predicted_streams[0]} 🎉")