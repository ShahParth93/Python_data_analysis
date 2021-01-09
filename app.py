from flask import Flask,render_template, request, redirect,session
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import FloatField, IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired, InputRequired, NumberRange

import joblib
import sklearn
import numpy as np
import pandas as pd

app = Flask(__name__)
Bootstrap(app)
app.config["SECRET_KEY"] = "happy new year 2021"


@app.route("/")
def index():
	return render_template('index.html')

@app.route("/<name>")
def welcoming_message(name):
	return "<h1>Hello World {}</h1>".format(name)


@app.route('/predict1', methods=['POST'])
def home():
    
    data1 = request.form['thal']
    data2 = request.form['cp']
    data3 = request.form['thalach']
    data4 = request.form['ca']
    data5 = request.form['oldpeak']

    
    model_load = joblib.load("./model_saved11")
    
    pred = model_load.predict([[data1,data2,data3,data4,data5]])


    return render_template('predict.html', data=pred)



if __name__ == '__main__':
	app.run(host='127.0.0.1',port=5000,debug=True)