import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# Flask
from flask import Flask, request, render_template, jsonify
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

from src.NucleiDetector import NucleiDetector

# Declare a flask app
app = Flask(__name__)

nucleiDetector = NucleiDetector()
print('Model loaded. Check http://127.0.0.1:5000/')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        nucleiDetector.load_images(*request.json)

        cmcount, nucleicount, pmask = nucleiDetector.predict()

        # Serialize the result, you can add additional fields
        return jsonify(cmcount=cmcount, nucleicount=nucleicount)
    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
