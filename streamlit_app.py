import streamlit as st
import pandas as pd


st.title("🎬 Movie Ratings Dashboard")

# File upload
movies_file = st.file_uploader("Upload movies.csv", type="csv")
ratings_file = st.file_uploader("Upload ratings.csv", type="csv")
tags_file = st.file_uploader("Upload tags.csv", type="csv")
df1 = pd.read_csv(movies_file)
df2 = pd.read_csv(ratings_file)
df3 = pd.read_csv(tags_file)


    