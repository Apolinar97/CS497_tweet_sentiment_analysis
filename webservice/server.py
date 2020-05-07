import os
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
import nltk
nltk.download('stopwords')
import re
from pre_process import validate_tweet, process_tweet
from nltk.corpus import stopwords
from flask import Flask, jsonify, render_template, render_template_string
from sklearn.preprocessing import LabelEncoder

sentiment_model = tf.keras.models.load_model('sentiment_model_reg.h5')
app = Flask(__name__, template_folder='templates')
app.config['DEBUG'] = True

with open('token.pkl','rb') as token_file:
  tkn = pickle.load(token_file)



@app.route('/')
def home():
  return render_template('home.html')

@app.route('/analyze/<string:text>', methods=['GET'])
def analyze(text):
  if(len(text) > 1):
    if(validate_tweet(text)):
      text = process_tweet(text)
      text_oh = tkn.texts_to_matrix(text, mode='binary')
      
      result = sentiment_model.predict(text_oh)
      result = pd.DataFrame(data=result)
      
      html_str = result.to_html()
      return html_str
    
    else:
      return '<h1> Error </h1>'


app.run()

#labels:
# [0,1,2]
#negative, positive, neutral respectively 