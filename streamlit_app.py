import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🎬 Movie Ratings Dashboard")

# File upload
movies_file = st.sidebar.file_uploader("Upload movies.csv", type="csv")
ratings_file = st.sidebar.file_uploader("Upload ratings.csv", type="csv")
tags_file = st.sidebar.file_uploader("Upload tags.csv", type="csv")


    