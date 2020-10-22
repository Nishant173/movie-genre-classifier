import time
import joblib
import numpy as np
import pandas as pd
import warnings

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

def get_timetaken_fstring(num_seconds):
    """Returns formatted-string of time elapsed, given the number of seconds (int) elapsed"""
    if num_seconds < 60:
        secs = num_seconds
        fstring_timetaken = f"{secs}s"
    elif 60 < num_seconds < 3600:
        mins, secs = divmod(num_seconds, 60)
        fstring_timetaken = f"{mins}m {secs}s"
    else:
        hrs, secs_remainder = divmod(num_seconds, 3600)
        mins, secs = divmod(secs_remainder, 60)
        fstring_timetaken = f"{hrs}h {mins}m {secs}s"
    return fstring_timetaken

def run_and_timeit(func):
    """
    Takes in function; then runs it, times it, and prints out the time taken.
    Parameters:
        - func (object): Object of the function you want to execute.
    """
    start = time.time()
    warnings.filterwarnings(action='ignore')
    func()
    end = time.time()
    timetaken_in_secs = int(np.ceil(end - start))
    timetaken_fstring = get_timetaken_fstring(num_seconds=timetaken_in_secs)
    print(f"\nDone! Time taken: {timetaken_fstring}")
    return None