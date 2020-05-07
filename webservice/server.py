import os
import pickle
import pandas as pd
import tensorflow as tf
import numpy as np

from pre_process import prepare_tweet

from flask import Flask, jsonify, render_template
from flask import request, abort
from sklearn.preprocessing import LabelEncoder

sentiment_model = tf.keras.models.load_model('sentiment_model_reg.h5')
app = Flask(__name__, template_folder='templates')
app.config['DEBUG'] = True

with open('token.pkl', 'rb') as token_file:
    tkn = pickle.load(token_file)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict/<string:text>', methods=['GET'])
def analyze_GET(text):
    try:
        text = prepare_tweet(text)
        text_oh = tkn.texts_to_matrix(text, mode='binary')

        result = sentiment_model.predict(text_oh)
        result = pd.DataFrame(data=result)
        html_str = result.to_html()
        return html_str

    except:
        abort(400)


app.run()

# labels:
# [0,1,2]
# negative, positive, neutral respectively
