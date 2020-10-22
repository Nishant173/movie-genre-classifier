from nltk.stem import PorterStemmer
import re

def remove_stopwords(text, stopwords_list):
    """Remove stopwords from corpus of text"""
    text = text.lower()
    words = text.split()
    words_without_stopwords = [word for word in words if word not in stopwords_list]
    return " ".join(words_without_stopwords)

def stem(text):
    """Perform stemming on corpus of text"""
    ps = PorterStemmer()
    text = text.lower()
    words = text.split()
    words_stemmed = [ps.stem(word=word) for word in words]
    return " ".join(words_stemmed)

def clean_text(text, stopwords_list):
    """
    Definition:
        Cleans corpus of text.
    Parameters:
        - text (str): Corpus of text
        - stopwords_list (list/set): List of stopwords
    Performs following actions:
        - Lowercases the text
        - Keeps only alphabetical characters
        - Removes stopwords
        - Stems all words
    """
    text = text.lower()
    text = re.sub(pattern='[^a-zA-Z]', repl=' ', string=text)
    text = remove_stopwords(text=text, stopwords_list=stopwords_list)
    text = stem(text=text)
    return text