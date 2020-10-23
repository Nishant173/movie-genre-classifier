from collections import namedtuple
from itertools import product
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings(action='ignore')

import config
import data_reader
import text_cleaner
import utils

def preprocess_data():
    """Preprocess train-test data from scratch and return Pandas DataFrame of the same"""
    df_movie_data = data_reader.get_movie_data()
    stopwords_list = list(set(stopwords.words('english')))
    df_movie_data['Plot'] = df_movie_data['Plot'].apply(text_cleaner.clean_text, stopwords_list=stopwords_list)
    utils.pickle_save(data_obj=df_movie_data, filepath=config.PATH_MOVIE_DATA_PREPROCESSED)
    return df_movie_data

def get_train_test_data():
    """Returns Pandas DataFrame of preprocessed train-test movie plot/genre data"""
    if config.LOAD_PREPROCESSED_DATA:
        try:
            df_movie_data = utils.pickle_load(filepath=config.PATH_MOVIE_DATA_PREPROCESSED)
        except FileNotFoundError:
            df_movie_data = preprocess_data()
    else:
        df_movie_data = preprocess_data()
    return df_movie_data

def train_model(df_movie_data):
    """
    Trains classification model on 100% of the training dataset, and returns dictionary
    containing certain info about said model.
    """
    mlb = MultiLabelBinarizer()
    X = df_movie_data['Plot'].values
    y = mlb.fit_transform(df_movie_data['Genre'])
    tfidf = TfidfVectorizer(max_df=0.85, max_features=7000) # Choose best hyperparams
    X = tfidf.fit_transform(raw_documents=X)
    log_reg = LogisticRegression()
    ovr_clf = OneVsRestClassifier(estimator=log_reg)
    ovr_clf.fit(X=X, y=y)
    dictionary_train_info = {
        'X_train': None,
        'X_test': None,
        'y_train': None,
        'y_test': None,
        'y_pred': None,
        'mlb': mlb,
        'tfidf': tfidf,
        'ovr_clf': ovr_clf,
    }
    return dictionary_train_info

def get_f1_score(thresh, dictionary_train_info):
    """Helper function that returns f1-score for model evaluation"""
    ovr_clf = dictionary_train_info['ovr_clf']
    X_test = dictionary_train_info['X_test']
    y_test = dictionary_train_info['y_test']
    y_pred_prob = ovr_clf.predict_proba(X=X_test)
    y_pred_new = (y_pred_prob >= thresh).astype(int)
    f1_score_by_thresh = f1_score(y_true=y_test, y_pred=y_pred_new, average="micro")
    return round(f1_score_by_thresh, 5)

def evaluate_model(dictionary_train_info):
    """
    Evaluates the classification model based on certain training info.
    Returns dictionary of the model's evaluation metrics.
    """
    f1_score_thresh50 = get_f1_score(thresh=0.5, dictionary_train_info=dictionary_train_info)
    f1_score_thresh30 = get_f1_score(thresh=0.3, dictionary_train_info=dictionary_train_info)
    f1_score_thresh25 = get_f1_score(thresh=0.25, dictionary_train_info=dictionary_train_info)
    f1_score_thresh20 = get_f1_score(thresh=0.2, dictionary_train_info=dictionary_train_info)
    dictionary_model_metrics = {
        'model_version': utils.get_current_timestamp(),
        'f1_score_thresh50': f1_score_thresh50,
        'f1_score_thresh30': f1_score_thresh30,
        'f1_score_thresh25': f1_score_thresh25,
        'f1_score_thresh20': f1_score_thresh20,
    }
    return dictionary_model_metrics

def get_runs(dictionary_hyperparams):
    """
    Takes in dictionary of hyperparams and returns list of runs wherein each run contains a unique
    combination hyperparams to be used for experimentation.
    NOTE: `dictionary_hyperparams` will be dict with keys = name of hyperparameter, and
    values = list of values of hyperparameter to try out.
    """
    run_tuple = namedtuple('Run', dictionary_hyperparams.keys())
    runs_list = []
    for value in product(*dictionary_hyperparams.values()):
        runs_list.append(run_tuple(*value))
    return runs_list

def try_and_log_various_hyperparams():
    """
    Tries various combinations of model hyperparams (based on config.HYPERPARAMS_TO_TRY)
    Saves CSV file containing info about various hyperparams tried (at config.PATH_HYPERPARAMS_TRIED)
    """
    df_movie_data = get_train_test_data()
    df_hyperparams_tried = pd.DataFrame()
    runs = get_runs(dictionary_hyperparams=config.HYPERPARAMS_TO_TRY)
    runs = pd.Series(data=runs).sample(len(runs)).tolist() # Shuffle order of runs
    for run in runs:
        max_df = run.max_df
        max_features = run.max_features
        mlb = MultiLabelBinarizer()
        X = df_movie_data['Plot']
        y = mlb.fit_transform(df_movie_data['Genre'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        tfidf = TfidfVectorizer(max_df=max_df, max_features=max_features)
        X_train = tfidf.fit_transform(raw_documents=X_train)
        X_test = tfidf.transform(raw_documents=X_test)
        log_reg = LogisticRegression()
        ovr_clf = OneVsRestClassifier(estimator=log_reg)
        ovr_clf.fit(X=X_train, y=y_train)
        y_pred = ovr_clf.predict(X=X_test)
        dictionary_train_info = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'mlb': mlb,
            'tfidf': tfidf,
            'ovr_clf': ovr_clf,
        }
        dict_model_eval_metrics = evaluate_model(dictionary_train_info=dictionary_train_info)
        df_temp_hyperparams = pd.DataFrame(data={
            'max_df': max_df,
            'max_features': max_features,
            'f1_score_thresh50': dict_model_eval_metrics['f1_score_thresh50'],
            'f1_score_thresh30': dict_model_eval_metrics['f1_score_thresh30'],
            'f1_score_thresh25': dict_model_eval_metrics['f1_score_thresh25'],
            'f1_score_thresh20': dict_model_eval_metrics['f1_score_thresh20'],
        }, index=[0])
        df_hyperparams_tried = pd.concat(objs=[df_hyperparams_tried, df_temp_hyperparams], axis=0, ignore_index=True, sort=False)
    f1_score_columns = ['f1_score_thresh50', 'f1_score_thresh30', 'f1_score_thresh25', 'f1_score_thresh20']
    df_hyperparams_tried['highest_f1_score'] = df_hyperparams_tried[f1_score_columns].max(axis=1)
    df_hyperparams_tried.sort_values(by='highest_f1_score', ascending=False, ignore_index=True, inplace=True)
    df_hyperparams_tried.to_csv(config.PATH_HYPERPARAMS_TRIED, index=False)
    return None

def execute_training_pipeline():
    """
    Executes training pipeline consisting of the following steps:
    - Get preprocessed training data
    - Train a classification model on 100% of training dataset
    - Save certain info about said model (to model-related pickle files)
    """
    print("Training the model...")
    df_movie_data = get_train_test_data()
    dictionary_train_info = train_model(df_movie_data=df_movie_data)
    utils.pickle_save(data_obj=dictionary_train_info['mlb'], filepath=config.PATH_MODEL_MLB)
    utils.pickle_save(data_obj=dictionary_train_info['tfidf'], filepath=config.PATH_MODEL_TFIDF)
    utils.pickle_save(data_obj=dictionary_train_info['ovr_clf'], filepath=config.PATH_MODEL_OVR_CLF)
    print("Model training done.")
    return None

if __name__ == "__main__":
    utils.run_and_timeit(func=execute_training_pipeline)