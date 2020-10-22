import pandas as pd
from nltk.corpus import stopwords
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
    Trains classification model and returns dictionary containing info about said model.
    Also saves the necessary model-related files aptly.
    """
    mlb = MultiLabelBinarizer()
    X = df_movie_data['Plot']
    y = mlb.fit_transform(df_movie_data['Genre'])
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Create features
    tfidf = TfidfVectorizer(max_df=0.8, max_features=10000)
    X_train = tfidf.fit_transform(raw_documents=X_train)
    X_test = tfidf.transform(raw_documents=X_test)
    # Build classifier
    lr = LogisticRegression()
    ovr_clf = OneVsRestClassifier(lr)
    ovr_clf.fit(X=X_train, y=y_train)
    y_pred = ovr_clf.predict(X=X_test)
    # Saving necessary model-related files
    utils.pickle_save(data_obj=mlb, filepath=config.PATH_MODEL_MLB)
    utils.pickle_save(data_obj=tfidf, filepath=config.PATH_MODEL_TFIDF)
    utils.pickle_save(data_obj=ovr_clf, filepath=config.PATH_MODEL_OVR_CLF)
    dictionary_train_info = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
    }
    return dictionary_train_info

def evaluate_model(ovr_clf, dictionary_train_info):
    """
    Evaluates the model based on it's classifier object and certain training info.
    Returns dictionary of the model's evaluation metrics.
    """
    thresh = 0.3
    X_test = dictionary_train_info['X_test']
    y_test = dictionary_train_info['y_test']
    y_pred = dictionary_train_info['y_pred']
    f1_score_thresh50 = f1_score(y_true=y_test, y_pred=y_pred, average="micro")
    y_pred_prob = ovr_clf.predict_proba(X=X_test)
    y_pred_new = (y_pred_prob >= thresh).astype(int)
    f1_score_thresh30 = f1_score(y_true=y_test, y_pred=y_pred_new, average="micro")
    dictionary_model_metrics = {
        'f1_score_thresh50': round(f1_score_thresh50, 5),
        'f1_score_thresh30': round(f1_score_thresh30, 5),
    }
    utils.pickle_save(data_obj=dictionary_model_metrics, filepath=config.PATH_MODEL_EVAL_METRICS)
    return dictionary_model_metrics

def execute_training_pipeline():
    df_movie_data = get_train_test_data()
    dictionary_train_info = train_model(df_movie_data=df_movie_data)
    ovr_clf = utils.pickle_load(filepath=config.PATH_MODEL_OVR_CLF)
    dictionary_model_metrics = evaluate_model(ovr_clf=ovr_clf, dictionary_train_info=dictionary_train_info)
    print(f"Model evaluation metrics: {dictionary_model_metrics}")
    return None

if __name__ == "__main__":
    print("Training the model...")
    utils.run_and_timeit(func=execute_training_pipeline)