import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from skimage import io
from tensorflow.keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template, Markup
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


# Define a flask app
app = Flask(__name__)

#app.config['SERVER_NAME'] = '0.0.0.0:5000'
# app.config.from_pyfile(config.cfg)

# List of Plants
MODEL_DICT = {
    "Apple": "Apple.h5",
    "Cherry": "Cherry.h5",
    "Corn": "Corn.h5",
    "Grape": "Grape.h5",
    "Peach": "Peach.h5",
    "Pepperbell": "Pepperbell.h5",
    "Potato": "Potato.h5",
    "Strawberry": "Strawberry.h5",
    "Tomato": "Tomato.h5"
}


def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32')
    x /= 255
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    dropdown_options = [k for k, v in MODEL_DICT.items()]
    return render_template('index.html', options=dropdown_options)


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        pname = request.form["plant_name"]
        model = tf.keras.models.load_model(MODEL_DICT[pname], compile=False)

        print('Model loaded!!')

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds)

        disease_class = {"Apple": ['Apple_scab', 'Black_rot', 'Cedar_apple_rust', 'Healthy'],
                         "Cherry": ['Powdery_mildew', 'Healthy'],
                         "Corn": ['Cercospora_leaf_spot Gray_leaf_spot', 'Common_rust_', 'Northern_Leaf_Blight', 'Healthy'],
                         "Grape": ['Black_rot', 'Esca_(Black_Measles)', 'Leaf_blight_(Isariopsis_Leaf_Spot)', 'Healthy'],
                         "Peach": ['Bacterial_spot', 'Healthy'],
                         "Pepperbell": ['Bacterial_spot', 'Healthy'],
                         "Potato": ['Early_blight', 'Late_blight', 'Healthy'],
                         "Strawberry": ['Leaf_scorch', 'Healthy'],
                         "Tomato": ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot', 'Spider_mites Two-spotted_spider_mite', 'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus', 'Healthy'],
                         }
        a = preds[0]
        ind = np.argmax(a)
        print('Prediction:', pname, disease_class[pname][ind])
        result = disease_class[pname][ind]
        print(result)
        return {
            "plant": pname,
            "status": result,
        }
    return None


if __name__ == '__main__':
    # app.run(port=5000, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    app.run()
