import streamlit as st

import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px 
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("data.csv")
genre_data = pd.read_csv('data_by_genres.csv')
year_data = pd.read_csv('data_by_year.csv')

def tlower(name):
    name = name.lower()
    return name

# data['name'] = data['name'].apply(tlower)

st.title("Music Recommender")

from yellowbrick.target import FeatureCorrelation

feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'loudness', 'speechiness', 'tempo', 'valence','duration_ms','explicit','key','mode','year']

X, y = data[feature_names], data['popularity']

# Create a list of the feature names
features = np.array(feature_names)

# Instantiate the visualizer
visualizer = FeatureCorrelation(labels=features)

plt.rcParams['figure.figsize']=(20,20)
visualizer.fit(X, y)

from collections import defaultdict
counts = defaultdict(int)
def get_decade(year, counts):
    period_start = int(year/10) * 10
    counts[str(period_start)] += 1
    return counts
    # return counts
for i in range(len(data)):
    counts = get_decade(data['year'][i], counts)
# st.write(counts)
xData = []
yData = []
for i in counts:
    xData.append(int(i))
    yData.append(counts[i])
lineG = pd.DataFrame(yData, xData)

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10))])
X = genre_data.select_dtypes(np.number)
cluster_pipeline.fit(X)
genre_data['cluster'] = cluster_pipeline.predict(X)

# Visualizing the Clusters with t-SNE

from sklearn.manifold import TSNE

tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
genre_embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
projection['genres'] = genre_data['genres']
projection['cluster'] = genre_data['cluster']

song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                ('kmeans', KMeans(n_clusters=20, 
                                verbose=False))
                                ], verbose=False)

X = data.select_dtypes(np.number)
number_cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
data['cluster_label'] = song_cluster_labels

# Visualizing the Clusters with PCA

from sklearn.decomposition import PCA

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = data['name']
projection['cluster'] = data['cluster_label']

def edaPage():
    st.subheader("Feature Correlation")
    st.image("feature correlation.jpg")

    st.subheader("Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, fmt='.2f')
    st.write(fig)

    st.subheader("Count Plot")
    st.line_chart(lineG, height = 300)

    st.subheader('features vs values')
    sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
    st.line_chart(year_data, x='year', y=sound_features)

    st.subheader("genre")
    top10_genres = genre_data.nlargest(10, 'popularity')
    fig = px.bar(top10_genres, x='genres', y=sound_features, height=750)
    st.plotly_chart(fig)

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=10))])
    X = genre_data.select_dtypes(np.number)
    cluster_pipeline.fit(X)
    genre_data['cluster'] = cluster_pipeline.predict(X)

    # Visualizing the Clusters with t-SNE

    from sklearn.manifold import TSNE

    tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, verbose=1))])
    genre_embedding = tsne_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
    projection['genres'] = genre_data['genres']
    projection['cluster'] = genre_data['cluster']

    song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                    ('kmeans', KMeans(n_clusters=20, 
                                    verbose=False))
                                    ], verbose=False)

    X = data.select_dtypes(np.number)
    number_cols = list(X.columns)
    song_cluster_pipeline.fit(X)
    song_cluster_labels = song_cluster_pipeline.predict(X)
    data['cluster_label'] = song_cluster_labels

    # Visualizing the Clusters with PCA

    from sklearn.decomposition import PCA

    pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2))])
    song_embedding = pca_pipeline.fit_transform(X)
    projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
    projection['title'] = data['name']
    projection['cluster'] = data['cluster_label']

    st.subheader("scatter plot")
    projection['genres'] = genre_data['genres']
    fig = px.scatter(projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
    st.write(fig)

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id= ['50d33ace53184c5984093acb0b2f8559'],
                                                           client_secret=['4b2d3832722a4f6f8a988ba20e1c6df2']))

def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)

from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import difflib

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']


def get_song_data(song, spotify_data):
    
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    
    except IndexError:
        return find_song(song['name'], song['year'])
        

def get_mean_vector(song_list, spotify_data):
    
    song_vectors = []
    
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict


def recommend_songs( song_list, spotify_data, n_songs=10):
    
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs+1][0])
    
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    # for i in metadata_cols:
    st.write(rec_songs[metadata_cols].to_dict(orient='records'))
    # return rec_songs[metadata_cols].to_dict(orient='records')

def musicRecommender():
    name = data['name']
    songName = st.selectbox("Select Song", name, index = 0)
    def findYear(name):
        if name == songName:
            Yr = data["year"]
            yield Yr
    years=data.loc[data['name'] == songName, 'year'].to_numpy()
    songYear = st.selectbox("Select Year", years, index = 0)
    nSongs = st.slider("Number of Songs", min_value = 1, max_value = 10, value = 5)
    recommend_songs([{'name': songName, 'year':songYear}],  data, nSongs)

def dataPage():
    st.subheader("Original Dataset")
    st.dataframe(data)
    st.write("data shape : ", data.shape)

def aboutUsPage():
    
    st.subheader("About us")
    st.write("This Project was done for the course Machine Learning Laboratory(20XD57)")
    st.write("Developed by: ")
    st.write("Aditya Ramanathan [20pd02]")
    st.write("Kartheepan G      [20pd11]")
    st.header("\n\n\t\tTHANK YOU\n")

page = {
    "Music Recommender" : musicRecommender,
    "EDA" : edaPage,
    "Data" : dataPage,
    "About Us" : aboutUsPage
}

pages = st.sidebar.selectbox("Select the Page : ", page.keys())
page[pages]()