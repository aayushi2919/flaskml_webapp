# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:55:35 2019

@author: aaybagga
"""

from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import NLPModel
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

app = Flask(__name__)
api = Api(app)

model = NLPModel()

clf_path = 'models/Classifier.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)

vec_path = 'models/TFIDFVectorizer.pkl'
with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    
    data=request.form.values()
    uq_vectorized = model.vectorizer_transform(data)
    prediction = model.predict(uq_vectorized)
    pred_proba = model.predict_proba(uq_vectorized)

    if prediction == 1:
        output = 'Yes'
    else:
        output = 'No'

    return render_template('index.html', prediction_text='FITARA applicable {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    