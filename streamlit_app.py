import streamlit as st

import pandas as pd

import plotly.express as px

import numpy as np

from scipy.stats import normaltest



st.title('Interactive Movie Ratings Dashboard')



# You might want to use a file uploader in Streamlit

# df = pd.read_csv('path_to_your_file.csv')  # Replace with actual file path



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

%config InlineBackend.figure_format = 'retina'
sns.set_context('talk')
import warnings
warnings.filterwarnings('ignore')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#pip install streamlit

df1 = pd.read_csv('movies.csv', encoding='latin1')

df2 = pd.read_csv('rating.csv')


df3 = pd.read_csv('tags.csv', encoding='latin1')

df1.head()

df2.head()

df3.head()

df1['movies.csv'] = 'movies'
df2['rating.csv'] = 'ratings'
df3['tags.csv'] = 'tags'

df = pd.merge(df1, df2, on='movieId', how='inner' )

df.head()

df = pd.merge(df, df3, on='movieId', how='inner')

df.head()

df.shape

df.duplicated().sum()

df = df.drop_duplicates()

num_ratings = df.groupby('movieId')['rating'].count().reset_index()
num_ratings.columns = ['movieId', 'num_ratings']

df = df.merge(num_ratings, on='movieId', how='left')
df.head()

df.shape

df = df.drop(columns=["movies.csv", "timestamp_x", "rating.csv", "userId_y", "timestamp_y", "tags.csv"])

df.head()

df.shape

df = df.drop_duplicates()

df['year'] = df['title'].str.extract(r'\((\d{4})\)').astype(float)

df = df[df['year'] >= 2009]

df.shape

df = df.sample(frac=0.1, random_state=42)

df.shape

df['first_genre'] = df['genres'].apply(lambda x: x.split('|')[0])

df.head()

df.shape

genres_options = [{'label': genres, 'value': genres} for genres in df['first_genre'].unique()]
#tag_options = [{'label': tag, 'value': tag} for tag in df['tag'].unique()]

movie_counts = df['title'].value_counts().head(5)

movie_counts

import dash
from dash import html, dcc
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output

#!pip install dash

movie_counts = df['title'].value_counts().head(5)

fig1 = px.bar(
    x=movie_counts.index,
    y=movie_counts.values,
    labels={'x': 'Movie Title', 'y': 'Count'},
    title="Top 5 Most Frequent Movies Overall"
)




import plotly.express as px

fig2 = px.histogram(
    df,
    x='rating',
    nbins=10,  # Number of bins (adjust as you like)
    title='Distribution of Movie Ratings',
    labels={'rating': 'Rating', 'count': 'Number of Ratings'},
    color_discrete_sequence=['skyblue']  # Optional: custom color
)



import plotly.express as px

df['year'] = df['year'].astype(int)
df = df.sort_values(by='year')

df["year"].unique()

all_genres = sorted(df['first_genre'].dropna().unique())

colors = px.colors.qualitative.Plotly
color_map = {genre: colors[i % len(colors)] for i, genre in enumerate(all_genres)}

fig3 = px.scatter(df, x="num_ratings", y="rating", animation_frame="year",
    animation_group="title", color="first_genre", hover_name="title",
    size_max=45,
    title="Movie Ratings vs. Popularity Over Time",
    category_orders={'first_genre': all_genres},
    color_discrete_map=color_map,
    range_x=[0, 50000]
)

fig3.show()

app = dash.Dash()

app.layout = html.Div([
    html.H1("Explore Movie Ratings & Trends", style = {'textAlign':'center'}),
    html.P("Use the filters below to explore top-rated movies, popular genres, and tag-based trends.", style = {'textAlign':'center'}),

    html.Label("Select Genre:"),
    dcc.Dropdown(id='genre-dropdown',options=genres_options, value=genres_options[0]['value']),

    html.Br(),
    html.Div(id='movie-output'),

    html.Label('Filter by Rating Range'),
    dcc.RangeSlider(
        min=1,
        max=5,
        step=0.5,
        marks={i: str(i) for i in range(1, 6)},
        value=[1, 5],
        id='rating-slider'
    ),

    dcc.Graph(id='graph-1', figure=fig1),
    dcc.Graph(id='graph-2', figure=fig2),
    dcc.Graph(id='graph-3', figure=fig3),
    dcc.Graph(id='graph-4'),
    dcc.Graph(id='graph-5'),
])


@app.callback(
    Output('movie-output', 'children'),
    Input('genre-dropdown', 'value'),
)
def update_movies(selected_genre):
    filtered = df[(df['first_genre'] == selected_genre)]
    if filtered.empty:
        return "No movies found for this selection."
    return html.Ul([html.Li(movie) for movie in filtered['title'].drop_duplicates()])



if __name__ == '__main__': 
    app.run(port=8052)

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
import panel as pn

%matplotlib inline

top_rated_movies = df.groupby('title')['num_ratings'].max().sort_values(ascending=False).head(10)


plt.figure(figsize=(10,6))
top_rated_movies.plot(kind='barh', color='skyblue')
plt.xlabel('Number of Ratings')
plt.title('Top 10 Movies with Most Ratings')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show();

top_rated_movies = df.groupby('title')['num_ratings'].sum().sort_values(ascending=False).head(10).reset_index()

top_rated_movies.head(20)

import pandas as pd
import hvplot.pandas  # Import this to enable hvplot functionality on pandas DataFrames
import panel as pn
pn.extension()

plot = top_rated_movies.hvplot.barh(x='title', y='num_ratings', height=400, width=600,
                               title='Top 10 Movies with Most Ratings', invert=True)
pn.panel(plot).servable()







unique_ratings = df.drop_duplicates(subset=['userId_x', 'movieId'])

# PANEL WIDGET
min_ratings_slider = pn.widgets.IntSlider(name='Min Ratings for Avg Rating Chart', start=10, end=500, step=10, value=50)

avg_rating_df = unique_ratings.groupby(['movieId', 'title']).agg(
    avg_rating=('rating', 'mean'),
    num_ratings=('num_ratings', 'max')  # Use max because it's same for all rows per movie
).reset_index()


min_ratings = 2000
filtered = avg_rating_df[avg_rating_df['num_ratings'] >= min_ratings]
top_avg = filtered.sort_values(by='avg_rating', ascending=False).head(10)

plt.figure(figsize=(10,6))
plt.barh(top_avg['title'], top_avg['avg_rating'], color='orange')
plt.xlabel('Average Rating')
plt.title(f'Top 10 Highest Rated Movies (min {min_ratings} ratings)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show();







fig = px.histogram(df, x='rating', nbins=5, title='Ratings Distribution')
st.plotly_chart(fig)

import plotly.express as px

xmin,xmax=min(df.num_ratings), max(df.num_ratings)
ymin,ymax=min(df.rating), max(df.rating)

fig=px.scatter(df, x="num_ratings", y="rating", 
              animation_group="genre", color="title", hover_name="genre",
              facet_col="title", width=1580, height=400, 
              log_x=True, size_max=45, 
              range_x=[xmin, xmax], range_y=[ymin, ymax])











#if __name__ == '__main__': 
   # app.run(port=8052