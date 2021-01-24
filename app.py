from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from IPython.display import FileLink
from imutils import paths
from fastai.basic_train import load_learner
from fastai.vision import open_image
from flask_cors import CORS,cross_origin
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(0)
import pickle
import random
import shutil
import cv2
import os

from flask_cors import CORS,cross_origin


app = Flask(__name__)
CORS(app, support_credentials=True)


# load the learner
learn = load_learner(path='./model', file = 'RF_model.pkl')
classes = learn.data.classes
#RFC_Model = pickle.load(open('./model/RF_model.pkl', 'rb'))


def predict_single(img_file):
    'function to take image and return prediction'
    test_image = cv2.imread(img_file)
    test_image = cv2.cvtColor(test_image, cv2.IMREAD_GRAYSCALE)
    test_image = cv2.resize(test_image, (224, 224))
    test_img = test_image.flatten().reshape(1, -1)
    

    RFC_pred = learn.predict(test_img)
    return {
        'category': RFC_pred
    }

# route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    return jsonify(predict_single(request.files['image']))

if __name__ == '__main__':
    app.run()
