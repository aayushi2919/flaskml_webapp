from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from model import NLPModel
#from modelRNN import NLPModel2
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
import sqlite3
import tensorflow as tf
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
#from keras.models import load_model
#from rnn_new import test
app = Flask(__name__)
api = Api(app)
import keras
import tensorflow as tf

#from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from tensorflow.python.keras.backend import set_session
#from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model



# IMPORTANT: models have to be loaded AFTER SETTING THE SESSION for keras! 
# Otherwise, their weights will be unavailable in the threads after the session there has been set


with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    
    sess = tf.compat.v1.Session()
    set_session(sess)
    model = load_model('models/model.h5')
    model._make_predict_function()
    graph = tf.get_default_graph()
    
'''
with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    model = tf.keras.models.load_model('models/model.h5')
    sess = tf.compat.v1.Session()
    model._make_predict_function()
    graph = tf.get_default_graph()'''
model1 = NLPModel()
import h5py
#model2 = NLPModel2()

clf_path= 'models/Classifier.pkl'
with open(clf_path, 'rb') as f:
    model1.clf = pickle.load(f)

vec_path = 'models/TFIDFVectorizer.pkl'
with open(vec_path, 'rb') as f:
    model1.vectorizer = pickle.load(f)

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
    uq_vectorized = model1.vectorizer_transform(data)
    prediction = model1.predict(uq_vectorized)
    pred_proba = model1.predict_proba(uq_vectorized)

    if prediction>=0.6:
        answer="Yes"
    else:
        answer="No"


    return render_template('index.html', prediction_text='FITARA applicable {}'.format(output))

@app.route('/predict2',methods=['POST'])
def predict2():
    text=request.form.values()
    
    max_words = 50000
    max_len = 150

    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(text)
    sequences = tok.texts_to_sequences(text)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
    text=np.array(sequences_matrix)
    
    


    global sess
    global graph
    with graph.as_default():
        set_session(sess)
        prediction=model.predict(text,steps=10)

    #my_prediction = my_model.test(text)

        #Get the probability of it being FITARA AFFECTED
    #prediction=my_prediction[0][1]
    if prediction > 0.5:
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
    
    
    
    