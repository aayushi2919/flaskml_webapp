# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 19:22:16 2019

@author: aaybagga
"""

import re
import nltk
import keras
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV
from sklearn import datasets, linear_model
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense, SimpleRNN
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
data = pd.read_csv('extract_combined.csv')
data2 = pd.read_csv('labels.csv')
df= pd.merge(data,data2)
yn = {'Yes': 1,'No': 0}   
df.is_fitara = [yn[i] for i in df.is_fitara] 

sw = stopwords.words('english')
listt=['http','is','the','www','a','_','-','are','for','any','be','no','of','by','this','as','with','in','yes','x','"','b','c','h','j','“','also','”','•','’']
sw.append(listt) 
np.array(sw)

stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

def tok_word(text):
  text=word_tokenize(text)
  return " ".join(text)

def rem_num(text):
  text = re.sub(r'\d+', '', text)
  return "".join(text)

def lower(text):
    text = [word.lower() for word in text.split()]
    return " ".join(text)

def stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    return " ".join(text)

def remove_punctuation(text):
    import string
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def stemming(text):    
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text)

def lemm(text):
  text=[lemmatizer.lemmatize(word) for word in text.split()]
  return " ".join(text)

data['text'] = data['text'].apply(rem_num)
data['text'] = data['text'].apply(tok_word)
data['text'] = data['text'].apply(lower)
data['text'] = data['text'].apply(stopwords)
data['text'] = data['text'].apply(remove_punctuation)
data['text'] = data['text'].apply(stemming)
data['text'] = data['text'].apply(lemm)

#model = tf.keras.layers.SimpleRNN()
max_words = 50000
max_len = 150

model =tf.keras.Sequential()
#model.add(layers.LSTM(return_sequences=True))
model.add(Embedding(max_words, 8, input_length=max_len))
#model.add(Embedding(64, activation='relu', input_length=max_len))
model.add(Flatten())
'''model.add(layers.Dense(256,name='FC1'))
model.add(layers.Dense(256,name='FC2'))
model.add(layers.Dense(1,name='out_layer'))'''
model.add(layers.Dense(1, activation='sigmoid'))
X_train, X_test, y_train, y_test = train_test_split(df.text, df.is_fitara, test_size=0.25,shuffle=True)
'''max_words = 50000
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
'''

tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.fit(np.array(sequences_matrix),y_train,batch_size=4,epochs=10,steps_per_epoch=10,validation_split=0.1)
model.save('models/model.h5')
'''
def test(X_test):  
  test_sequences = tok.texts_to_sequences(X_test)
  test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
  return test_sequences_matrix
  
'''


