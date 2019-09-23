from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import NLPModel
from modelRNN import NLPModel2
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import sqlite3
import tensorflow as tf
from keras.models import Sequential

app = Flask(__name__)
api = Api(app)

model = NLPModel()
#model2 = NLPModel2()

clf_path= 'models/Classifier.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)

vec_path = 'models/TFIDFVectorizer.pkl'
with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)

parser = reqparse.RequestParser()
parser.add_argument('query')


@app.route('/')
def home():
    return render_template('index.html')

@app.route("/saveDetails",methods = ["POST","GET"])  
def saveDetails():  
    msg = "msg"  
    if request.method == "POST":  
        try:  
            doc_text = request.form["doc_text"]  
            with sqlite3.connect("docs.db") as con:  
                cur = con.cursor()  
                cur.execute("INSERT into tab(id,doc_text) values (?,?)",(id,doc_text))  
                
                con.commit()  
                msg = "Document successfully Added"  
        except:  
            con.rollback()  
            msg = "We can not add the document to the list"  
        finally:  

            return render_template("success.html",msg = msg)  
            con.close()  

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

@app.route('/predict2',methods=['POST'])
def predict2():

    
    text=request.form.values()
    self.model = Sequential()
    model.add(layers.Embedding(max_words, 64))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(256,name='FC1'))
    model.add(layers.Dense(256,name='FC2'))
    model.add(layers.Dense(1,name='out_layer'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    new_model = tf.keras.models.load_model('model.h5')
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(text)
    word_index = tokenizer.word_index
    #   print(len(word_index))

    X = tokenizer.texts_to_sequences(text)                         #Tokenize the dataset
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

        #Make prediction using the pre-trained model
    my_prediction = new_model.predict(X)

        #Get the probability of it being FITARA AFFECTED
    prediction=my_prediction[0][1]
    if prediction == 1:
        output = 'Yes'
    else:
        output = 'No'

    return render_template('index.html', prediction_text='FITARA applicable {}'.format(output))

@app.route("/view")  
def view():  
    con = sqlite3.connect("docs.db")  
    con.row_factory = sqlite3.Row  
    cur = con.cursor()  
    cur.execute("select * from tab")  
    rows = cur.fetchall()  
    return render_template("view.html",rows = rows)  


if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    