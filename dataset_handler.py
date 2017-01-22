import numpy as np
import os
import pandas as pd

genres = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western"
]

class DatasetHandler(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
    
    def ids2titles(self, ids):
        return [self.id_to_title[movieId] for movieId in ids]
    
    def indices2ids(self, indices):
        return [self.movie_index_to_movie_id[index] for index in indices]
    
    def id2index(self, movieId):
        return self.movie_index_to_movie_id.index(movieId)
    
    def movie_vector2genres(self, movie_vector):
        return [self.feature_index2genre(i) for i, x in enumerate(movie_vector) if x == 1]
    
    def feature_index2genre(self, feature_index):
        return genres[feature_index]

    def load_movies(self):
        movies_frame = pd.read_csv(os.path.join(self.dataset_path, "movies.dat"), names=["movieId", "title", "genres"], sep="::", engine="python")
        self.id_to_title = {}
        self.movie_index_to_movie_id = []
        movies_vectors = []
        for _, row in movies_frame.iterrows():
            genres_list = row["genres"].split("|")
            self.id_to_title[int(row["movieId"])] = row["title"]
            self.movie_index_to_movie_id.append(int(row["movieId"]))
            movies_vectors.append(np.array([1 if genre in genres_list else 0 for genre in genres]))
        return np.array(movies_vectors)

    def load_users_ratings(self):
        ratings_frame = pd.read_csv(os.path.join(self.dataset_path, "ratings.dat"), names=["userId", "movieId", "rating", "timestamp"], sep="::", engine="python")
        users_ratings = {}
        for _, row in ratings_frame.iterrows():
            if int(row["userId"]) not in users_ratings:
                users_ratings[int(row["userId"])] = {}
            users_ratings[int(row["userId"])][int(row["movieId"])] = row["rating"]
        return users_ratings
