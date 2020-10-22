import json
import pandas as pd
import config

def get_genre_list(json_obj):
    """Gets list of genres from JSON object"""
    return list(json.loads(json_obj).values())

def get_movie_genre_data():
    """Gets DataFrame containing movie-genre data"""
    dict_rename_mapper = {
        0: 'MovieId',
        2: 'MovieName',
        8: 'Genre',
    }
    df_movie_genre_data = pd.read_csv(f"{config.PATH_MOVIE_GENRE_DATA}", sep='\t', header=None)
    df_movie_genre_data.rename(dict_rename_mapper, axis=1, inplace=True)
    df_movie_genre_data = df_movie_genre_data.loc[:, list(dict_rename_mapper.values())]
    df_movie_genre_data['MovieId'] = df_movie_genre_data['MovieId'].astype(str)
    df_movie_genre_data['Genre'] = df_movie_genre_data['Genre'].apply(get_genre_list)
    return df_movie_genre_data

def get_movie_plot_data():
    """Gets DataFrame containing movie-plot data"""
    dict_rename_mapper = {
        0: 'MovieId',
        1: 'Plot',
    }
    df_movie_plot_data = pd.read_csv(f"{config.PATH_MOVIE_PLOT_DATA}", sep='\t', header=None)
    df_movie_plot_data.rename(dict_rename_mapper, axis=1, inplace=True)
    df_movie_plot_data['MovieId'] = df_movie_plot_data['MovieId'].astype(str)
    return df_movie_plot_data

def get_movie_data():
    """Gets Pandas DataFrame of movie-genre and movie-plot data"""
    df_movie_genre = get_movie_genre_data()
    df_movie_plot = get_movie_plot_data()
    df_movie_data = pd.merge(left=df_movie_genre, right=df_movie_plot, on='MovieId', how='inner')
    df_movie_data.dropna(axis=0, how='any', subset=['Genre', 'Plot'], inplace=True)
    df_movie_data.reset_index(drop=True, inplace=True)
    return df_movie_data