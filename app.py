from flask import Flask, render_template, jsonify
from sklearn.externals import joblib


app = Flask(__name__)

clf = joblib.load("data/digits_cls.pkl")


@app.route('/')
def index():
    """
    Uses Flask's Jinja2 template renderer to generate the html
    """
    return render_template('index.html')

@app.route('/predict/<image_vector>')
def predict(image_vector):

    arr = image_vector.split(",")
    for i in range(len(arr)):
        arr[i] = int(arr[i])
    print(len(image_vector.split(",")))
    prediction = clf.predict(arr)
    print(prediction)
    return jsonify(result=str(prediction))

if __name__ == '__main__':
    app.run(debug=True)
