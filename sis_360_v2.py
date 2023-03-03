# -*- coding: utf-8 -*-
"""SIS_360_v2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lUdAn2-JuF9prd6MMo0lR_jP_zih-ESt
"""

#establecemos  la conexion a drive para poder cargar los datos y guardar el modelo y el peso
from google.colab import drive
drive.mount('/content/drive')

!pip install tensorflow-addons

#librerias necesarias para que funciones la CNN
import sys
import os
from keras.preprocessing.image import ImageDataGenerator  
from keras.layers import Dropout
from keras import backend as K
from keras import optimizers
from keras.models import Sequential
from keras.layers import Activation, Dense, MaxPooling2D, Convolution2D, Flatten, Dropout
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

#declaramos los optimizadores a usar 
radam = tfa.optimizers.RectifiedAdam()
ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)

device_list = tf.test.gpu_device_name()

device_list

if device_list != '/device:GPU:0':
  raise SystemError('GPU device not found')
print ('Found GPU at: {}'.format(device_list))

#funciones para las métricas  que vamos a usar 

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
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# aqui matamos/ cerramos la sesión en caso de que estuviese trabajando en segundo plano
K.clear_session()

#cargamos los set de datos 
data_entrenamiento = ('/content/drive/MyDrive/SIS360/entrenamiento')
data_validacion = ('/content/drive/MyDrive/SIS360/validacion')

# variables con las que vamos a trabajar para poder entrenar la CNN
epocas= 55
longitud, altura = 150, 150
batch_size = 32
pasos_validacion = 300
clases = 4
learning_rate = 0.0005

# establecemos como vamos a trabajar para poder manipular las imagenes
entrenamiento_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True)

# establecemos el como va cargar los datos y el modo de clase con el que vamos a entrenar 
validacion_datagen = ImageDataGenerator(rescale=1. / 255)

imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size=(altura, longitud),
    batch_size = batch_size,
    class_mode= 'categorical' 
)

# lo mismo que el anterior pero para los datos de validación 
imagen_validacion = validacion_datagen.flow_from_directory(
    data_validacion,
    target_size = (altura, longitud),
    batch_size= batch_size,
    class_mode= 'categorical'
)

# empezamos a trabajar con las convoluciones cargando parametros 

cnn = Sequential()

cnn.add(keras.layers.Conv2D(
            512,
            (3, 3),
            padding="same", input_shape=(longitud, altura, 3),
            activation=keras.layers.LeakyReLU()
        ))


cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.AveragePooling2D())

cnn.add(keras.layers.Conv2D(
            256,
            (3, 3),
            padding="same", input_shape=(longitud, altura, 3),
            activation=keras.layers.LeakyReLU()
        ))

cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.AveragePooling2D())

cnn.add(Flatten())
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Dense(512, activation=keras.layers.LeakyReLU()))
cnn.add(keras.layers.BatchNormalization())
cnn.add(keras.layers.Dense(256, activation=keras.layers.LeakyReLU()))
cnn.add(keras.layers.BatchNormalization())
cnn.add(Dropout(0.4))
cnn.add(Dense(clases, activation='softmax'))

#compila la cnn con los parametros establecidos y nos da el porcentaje de precisión
cnn.compile(loss = 'categorical_crossentropy',
            optimizer = ranger,
            metrics = ["accuracy", f1_m, precision_m, recall_m])
cnn.summary()

#entrenamos el algoritmo y lo alimentamos con el entrenamiento
cnn.fit(
    imagen_entrenamiento,
    epochs=epocas,
    validation_data=imagen_validacion,
    validation_steps=pasos_validacion)

#guardamos en una carpeta los pesos y la estructura del modelo
target_dir = ('/content/drive/MyDrive/SIS360/modelo')
if not os.path.exists(target_dir):
  os.mkdir(target_dir)
cnn.save('/content/drive/MyDrive/SIS360/modelo/modelo.h5')
cnn.save_weights('/content/drive/MyDrive/SIS360/modelo/pesos.h5')

# se instala tensorflowjs para exportar el modelo y poderlo usar en una página web
!pip install tensorflowjs

# crear carpeta de salida para exportar a js

target_dir1 = ('/content/drive/MyDrive/carpeta_salida')
if not os.path.exists(target_dir1):
  os.mkdir(target_dir1)

# realizamos la exportación a la carpeta de salida 
!tensorflowjs_converter --input_format keras /content/drive/MyDrive/SIS360/modelo/modelo.h5 /content/drive/MyDrive/carpeta_salida

#confirmamos que se haya realizado

!ls carpeta_salida