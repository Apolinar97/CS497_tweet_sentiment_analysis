import os
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np
import nltk
nltk.download('stopwords')
import re

from nltk.corpus import stopwords
from flask import Flask, jsonify, render_template, render_template_string
from sklearn.preprocessing import LabelEncoder

sentiment_model = tf.keras.models.load_model('sentiment_model_reg.h5')
app = Flask(__name__, template_folder='templates')
app.config['DEBUG'] = True

with open('token.pkl','rb') as token_file:
  tkn = pickle.load(token_file)

def norm_tweet(tweet):
  tweet = tweet.lower()
  tweet = re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', tweet)
  tweet = re.sub('[^a-z]+', ' ', tweet)
  return tweet

def remove_stopwords_for_analysis(tweet):
  stop_words = stopwords.words('english')
  white_list = ['no', 'not']
  words_tweet = tweet.split()
  clean_tweet = [word for word in words_tweet if(word not in stop_words or word in white_list)]
  return ' '.join(clean_tweet)


def process_tweet(tweet):
  tweet = norm_tweet(tweet)
  return remove_stopwords_for_analysis(tweet)

def validate_tweet(tweet):
  if (not tweet or not isinstance(tweet,str)):
    return False
  else:
    return True


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