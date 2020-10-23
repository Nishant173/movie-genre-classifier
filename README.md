# movie-genre-classifier
Classifies movie genre based on text corpus of the plot of the movie (Multi-label classification)

## Installation
- You can install dependencies with `pip install -r requirements.txt` from the root directory

## Usage
- Use the `config.py` file in the `src` directory to change configuration settings, when necessary.
- Run `train.py` from the `src` directory to train the model OR to explore model hyperparameters.
- In order to make predictions, use the `predict_genres_from_plot` function from the `src/predict.py` file.
```python
from predict import predict_genres_from_plot
from utils import prettify_genres

tuple_genres = predict_genres_from_plot(text="The plot of some movie") # Returns tuple of possible genre/s
print(prettify_genres(tuple_genres=tuple_genres))
```

### Hyperparameter exploration and logging
You can log hyperparameters and evaluation metrics by going to `src/config.py` and setting `MODE = 'explore_hyperparams'`
![Hyperparameter exploration and logging](images/hyperparam_runs.jpeg)

### Training data
This is what the training data looks like
![Training data](images/training_data.jpeg)