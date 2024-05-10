from .Model import Model

import numpy as np
import tensorflow as tf    
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Activation, LeakyReLU

class DensenetModel(Model):
    def __init__(self, name, classes, batch_size, epochs, input_shape, save_dir):
        super().__init__(name, classes, batch_size, epochs, input_shape, save_dir)

    def build(self):
        model = tf.keras.applications.DenseNet121(
                    weights=None, include_top=False, input_tensor=tf.keras.Input(shape=self.input_shape))

        x = Flatten()(model.output)
        output = Dense(self.classes)(x)
        model = keras.Model(inputs=model.input, outputs=[output, model.get_layer('conv3_block12_concat').output])

        self.model = model
