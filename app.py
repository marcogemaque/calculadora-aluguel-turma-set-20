import flask
import pickle
from flask import render_template, Flask
import pandas as pd
import numpy as np
from math import exp

#Create the Flask app
app = Flask(__name__, template_folder="templates")

#Set up the app
@app.route('/', methods=['GET','POST'])
def main():
    if flask.request.method == "GET":
        return flask.render_template("home.html")

    if flask.request.method == "POST":
        rooms = flask.request.form['rooms']
        area = flask.request.form['area']
        area = float(area)
        area_fix = np.log(area)
        bairro = flask.request.form['bairro']
        df = pd.read_csv('./static/db.csv',sep=';',encoding='latin')
        dimensions = df['Hood'].unique()
        objs = enumerate(dimensions, start=2)
        size = dimensions.shape[0]
        size += 2
        predict_obj = np.zeros((1,size))
        predict_obj[0,0] = area_fix
        predict_obj[0,1] = rooms
        for count, obj in objs:
            if bairro == obj:
                coord = count
        predict_obj[0,coord] = 1

        model = pickle.load(open('./static/tree_model.pkl', 'rb'))
        pred = model.predict(predict_obj)
        return 'Rent is {}'.format(exp(pred))

if __name__ == "__main__":
    app.run(debug=True)