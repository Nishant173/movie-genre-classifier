PATH_MOVIE_GENRE_DATA = "../data/movie_metadata.tsv"
PATH_MOVIE_PLOT_DATA = "../data/plot_summaries.tsv"
PATH_MOVIE_DATA_PREPROCESSED = "../data/df_movie_data_preprocessed.pkl"
LOAD_PREPROCESSED_DATA = True # Set to False if you want to preprocess the raw data from scratch

PATH_MODEL_MLB = "../model_files/mlb_obj.pkl"
PATH_MODEL_TFIDF = "../model_files/tfidf_obj.pkl"
PATH_MODEL_OVR_CLF = "../model_files/ovr_clf_obj.pkl"

PATH_HYPERPARAMS_TRIED = "../model_files/hyperparams_tried.csv"

# Hyperparams that need to be tried out for the model (before selecting the best set of hyperparams)
HYPERPARAMS_TO_TRY = {
    'max_df': [0.65, 0.7, 0.75, 0.8, 0.85],
    'max_features': [7000, 8000, 9000, 10000, 11000],
}