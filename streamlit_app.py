import streamlit as st
import pandas as pd
import numpy as np

st.title("🎬 Movie Ratings Dashboard")

# File upload
movies_file = st.file_uploader("Upload movies.csv", type="csv")
ratings_file = st.file_uploader("Upload ratings.csv", type="csv")
tags_file = st.file_uploader("Upload tags.csv", type="csv")
df1 = pd.read_csv(movies_file, encoding ='latin1')
df2 = pd.read_csv(ratings_file)
df3 = pd.read_csv(tags_file, encoding ='latin1')

df = pd.merge(df1, df2s, on='movieId')
df = pd.merge(df, df3, on='movieId', how='left')
df.drop_duplicates(inplace=True)

# Create additional fields
df['year'] = df['title'].str.extract(r'\((\d{4})\)').astype(float)
df = df[df['year'] >= 2009]
df['first_genre'] = df['genres'].apply(lambda x: x.split('|')[0])
df['num_ratings'] = df.groupby('title')['rating'].transform('count')

# ✅ Now safe to use df
st.write("Sample of merged data:")
st.dataframe(df.head())

    