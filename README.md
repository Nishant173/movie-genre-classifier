# movie-genre-classifier
Classifies movie genre based on text corpus of the plot of the movie (Multi-label classification)

## Installation
- You can install dependencies with `pip install -r requirements.txt` from the root directory

## Usage
- Open terminal in the `src` directory and use the following snippet
```python
from predict import predict_genres_from_plot
from utils import prettify_genres

tuple_genres = predict_genres_from_plot(text="The plot of some movie") # Returns tuple of possible genre/s
print(prettify_genres(tuple_genres=tuple_genres))
```
- Use the `config.py` file in the `src` directory to change configuration settings, when necessary.