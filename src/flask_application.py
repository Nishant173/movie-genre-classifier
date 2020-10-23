from flask import Flask, render_template, jsonify, request
from predict import predict_genres_from_plot
from utils import prettify_genres

app = Flask(__name__)

@app.route(rule='/', methods=['GET'])
def home():
    return "<h1>This is the Home page. Please go to '/predict_genres'</h1>"

@app.route(rule='/predict_genres', methods=['GET', 'POST'])
def predict_genres():
    if request.method == 'POST':
        text = request.form.get('movie-plot', default="")
        tuple_genres = predict_genres_from_plot(text=text)
        message = prettify_genres(tuple_genres=tuple_genres)
        response = {"message": message, "status_code": 200}
        return jsonify(response), 200
    return render_template(template_name_or_list='prediction_form.html')

if __name__ == "__main__":
    app.run(debug=True)