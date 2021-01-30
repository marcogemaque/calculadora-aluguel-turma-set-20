import flask
import pickle
from flask import render_template, Flask
import pandas as pd

#Create the Flask app
app = Flask(__name__, template_folder="templates")

#Set up the app
@app.route('/', methods=['GET','POST'])
def main():
    if flask.request.method == "GET":
        return flask.render_template("home.html")

    if flask.request.method == "POST":
        room = flask.request.form['rooms']
        area = flask.request.form['area']
        bairro = flask.request.form['bairro']
        home_object = np.array([area,rooms,bairro])
        model = pickle.load(open("./static/pickle_xgb_Model.pkl", 'rb'))

if __name__ == "__main__":
    app.run(debug=True)