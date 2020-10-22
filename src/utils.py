import joblib
import pandas as pd

def pickle_load(filepath):
    """Loads data from pickle file, via joblib module"""
    data_obj = joblib.load(filename=filepath)
    return data_obj

def pickle_save(data_obj, filepath):
    """Stores data as pickle file, via joblib module"""
    joblib.dump(value=data_obj, filename=filepath)
    return None

def get_genre_value_count(data):
    """
    Takes in movie-genre DataFrame, and returns dictionary of value-counts of genres.
    Expects DataFrame with the column 'Genre' wherein each element in said column
    contains a list of all genres for a movie.
    """
    genre_occurrences = []
    for genres_by_movie in data['Genre'].tolist():
        genre_occurrences.extend(genres_by_movie)
    return pd.Series(data=genre_occurrences).value_counts().to_dict()

def prettify_genres(tuple_genres):
    """Takes in tuple of genre/s and returns string of pretty-printed genres"""
    if not isinstance(tuple_genres, tuple):
        raise TypeError("Expected tuple, but got {}".format(type(tuple_genres)))
    if len(tuple_genres) == 0:
        prettified_genres = "Could not predict genre"
    elif len(tuple_genres) == 1:
        prettified_genres = f"Genre is {tuple_genres[0]}"
    else:
        genres_commafied = ", ".join(tuple_genres)
        prettified_genres = f"Genres are {genres_commafied}"
    return prettified_genres