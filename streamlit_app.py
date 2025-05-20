import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🎬 Movie Ratings Dashboard")

# File upload
movies_file = st.sidebar.file_uploader("Upload movies.csv", type="csv")
ratings_file = st.sidebar.file_uploader("Upload ratings.csv", type="csv")
tags_file = st.sidebar.file_uploader("Upload tags.csv", type="csv")

# Only process when all three files are uploaded
if movies_file and ratings_file and tags_file:
    # Load data
    df_movies = pd.read_csv(movies_file, encoding='latin1')
    df_ratings = pd.read_csv(ratings_file)
    df_tags = pd.read_csv(tags_file, encoding='latin1')

    # Merge and prepare
    df = pd.merge(df_movies, df_ratings, on='movieId')
    df = pd.merge(df, df_tags, on='movieId', how='left')
    df.drop_duplicates(inplace=True)

    # Create additional fields
    df['year'] = df['title'].str.extract(r'\((\d{4})\)').astype(float)
    df = df[df['year'] >= 2009]
    df['first_genre'] = df['genres'].apply(lambda x: x.split('|')[0])
    df['num_ratings'] = df.groupby('title')['rating'].transform('count')

    # ✅ Now safe to use df
    st.write("Sample of merged data:")
    st.dataframe(df.head())