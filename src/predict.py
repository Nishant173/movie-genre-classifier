from nltk.corpus import stopwords
import config
import text_cleaner
import utils

def predict_genres_from_plot(text):
    """
    Predicts movie-genre from corpus of text of movie-plot.
    Returns tuple containing genre/s of movie-plot given.
    """
    # Load model-related files
    mlb = utils.pickle_load(filepath=config.PATH_MODEL_MLB)
    tfidf = utils.pickle_load(filepath=config.PATH_MODEL_TFIDF)
    ovr_clf = utils.pickle_load(filepath=config.PATH_MODEL_OVR_CLF)
    # Make the prediction and return tuple of predicted genre/s
    stopwords_list = list(set(stopwords.words('english')))
    text = text_cleaner.clean_text(text=text, stopwords_list=stopwords_list)
    dtm = tfidf.transform([text])
    pred = ovr_clf.predict(X=dtm)
    tuple_genres = mlb.inverse_transform(yt=pred)[0]
    return tuple_genres