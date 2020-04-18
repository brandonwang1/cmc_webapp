import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Some utilites
import numpy as np
from util import base64_to_pil

from src.NucleiDetector import NucleiDetector

# Declare a flask app
app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5000/')


nucleiDetector = NucleiDetector()
nucleiDetector.load_images()

def model_predict(img, model):
    img = img.resize((224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/results', methods=['GET'])
def results():
    # Main page
    return render_template('results.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    nucleiDetector.predict()
    # if request.method == 'POST':
    #     # Get the image from post request
    #     img = base64_to_pil(request.json)
    #
    #     # Save the image to ./uploads
    #     # img.save("./uploads/image.png")
    #
    #     # Make prediction
    #     preds = model_predict(img, model)
    #
    #     # Process your result for human
    #     pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
    #     pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
    #
    #     result = str(pred_class[0][0][1])               # Convert to string
    #     result = result.replace('_', ' ').capitalize()
    #
    #     # Serialize the result, you can add additional fields
    #     return jsonify(result=result, probability=pred_proba)
    results()
    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
