import streamlit as st
import pandas as pd
import plotly.express as px

# Set page config
st.set_page_config(page_title="Movie Ratings Dashboard", layout="wide")

# Title
st.title("🎬 Interactive Movie Ratings Dashboard")

# Load datasets
st.sidebar.header("Upload Your Datasets")

movies_file = st.sidebar.file_uploader("Upload movies.csv", type=["csv"])
ratings_file = st.sidebar.file_uploader("Upload ratings.csv", type=["csv"])
tags_file = st.sidebar.file_uploader("Upload tags.csv", type=["csv"])

if movies_file and ratings_file and tags_file:
    df1 = pd.read_csv(movies_file, encoding='latin1')
    df2 = pd.read_csv(ratings_file)
    df3 = pd.read_csv(tags_file, encoding='latin1')

    # Merge datasets
    df = pd.merge(df1, df2, on='movieId', how='inner')
    df = pd.merge(df, df3, on='movieId', how='inner')
    df.drop_duplicates(inplace=True)

    # Preprocessing
    df['year'] = df['title'].str.extract(r'\((\d{4})\)').astype(float)
    df = df[df['year'] >= 2009]
    df = df.sample(frac=0.1, random_state=42)
    df['first_genre'] = df['genres'].apply(lambda x: x.split('|')[0])

    # Top 5 Most Frequent Movies
    movie_counts = df['title'].value_counts().head(5)
    fig1 = px.bar(
        x=movie_counts.index,
        y=movie_counts.values,
        labels={'x': 'Movie Title', 'y': 'Count'},
        title="Top 5 Most Frequent Movies Overall"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # Ratings Histogram
    fig2 = px.histogram(
        df, x='rating', nbins=10,
        title='Distribution of Movie Ratings',
        labels={'rating': 'Rating', 'count': 'Number of Ratings'},
        color_discrete_sequence=['skyblue']
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Animated Scatter Plot
    df['year'] = df['year'].astype(int)
    all_genres = sorted(df['first_genre'].dropna().unique())
    colors = px.colors.qualitative.Plotly
    color_map = {genre: colors[i % len(colors)] for i, genre in enumerate(all_genres)}

    fig3 = px.scatter(
        df, x="rating", y="num_ratings", animation_frame="year", animation_group="title",
        color="first_genre", hover_name="title", size_max=45,
        title="Movie Ratings vs. Popularity Over Time",
        category_orders={'first_genre': all_genres},
        color_discrete_map=color_map,
        range_x=[1, 5], range_y=[0, df['num_ratings'].max()]
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Genre filter
    selected_genre = st.selectbox("Select a Genre", all_genres)
    filtered_df = df[df['first_genre'] == selected_genre]
    st.write(f"🎥 Movies in genre **{selected_genre}**", filtered_df[['title', 'rating']].drop_duplicates())
else:
    st.warning("Please upload all three datasets (movies.csv, rating.csv, tags.csv) to continue.")





