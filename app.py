import numpy as np
import os
import tensorflow as tf
import tensorflow_addons as tfa
from keras.models import load_model
from keras import backend as K
from flask import Flask, render_template, request, redirect, url_for, jsonify

app = Flask(__name__)

radam = tfa.optimizers.RectifiedAdam()
ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

longitud, altura = 150, 150

MODELO = "modelo/modelo.h5"
PESOS_MODELO = "modelo/pesos.h5"


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


cnn = load_model(
    MODELO,
    custom_objects={"f1_m": f1_m, "recall_m": recall_m, "precision_m": precision_m},
)

cnn.load_weights(PESOS_MODELO)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        return redirect(url_for("index"))

    return render_template("index.html")


@app.route("/contacto", methods=["GET", "POST"])
def contacto():
    if request.method == "POST":
        return redirect(url_for("index"))

    return render_template("contacto.html")


@app.route("/acerca", methods=["GET", "POST"])
def acerca():
    if request.method == "POST":
        return redirect(url_for("index"))

    return render_template("acerca.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        labels = ["Downy mildew", "Fresh Leaf", "Gray mold", "Leaf scars"]
        # Obtiene la imagen enviada por el usuario
        image = request.files["image"]
        filename = image.filename
        filepath = os.path.join("static/imagenes/", filename)
        image.save(filepath)
        
        # Convierte la imagen en un tensor para poder realizar la predicción
        image = tf.io.read_file(filepath)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [150, 150])
        image = np.array(image, dtype=np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        # Realiza la predicción
        val = cnn.predict(image)

        # Convierte la predicción en una cadena para poder enviarla a la página HTML
        prediction = labels[np.argmax(val, axis=-1)[0]]
        prediction = str(prediction)

        return render_template("prediccion.html", prediction=prediction, image_url=filepath)

    return render_template("prediccion.html", prediction="", image_url="")


@app.route("/recomendaciones", methods=["GET", "POST"])
def recomendaciones():
    if request.method == "POST":
        return redirect(url_for("index"))

    return render_template("recomendaciones.html")


if __name__ == "__main__":
    app.run(debug=True)

# comando para correr el servidor
# python -u "D:\modular\servidor_SIS\app.py" python3 -m flask run
