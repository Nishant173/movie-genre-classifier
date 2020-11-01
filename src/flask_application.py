from flask import Flask, render_template, request
from predict import predict_genres_from_plot
from utils import prettify_genres

app = Flask(__name__)

@app.route(rule='/', methods=['GET'])
def home():
    return render_template('home.html')

@app.route(rule='/predict-genres', methods=['GET', 'POST'])
def predict_genres():
    if request.method == 'POST':
        text = request.form.get('movie-plot', default="")
        tuple_genres = predict_genres_from_plot(text=text)
        prediction_msg = prettify_genres(tuple_genres=tuple_genres)
        return render_template('prediction_output.html', prediction_msg=prediction_msg)
    return render_template('prediction_form.html')

if __name__ == "__main__":
    app.run(debug=True)