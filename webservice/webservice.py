import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder

sentiment_model = tf.keras.models.load_model('sentiment_model_reg.h5')
