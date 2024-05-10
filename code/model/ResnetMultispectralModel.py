from .Model import Model

import numpy as np
import tensorflow as tf    
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Activation, LeakyReLU

class ResnetMultispectralModel(Model):
    def __init__(self, name, classes, batch_size, epochs, input_shape, save_dir):
        super().__init__(name, classes, batch_size, epochs, input_shape, save_dir)
    
    def build(self):
        inputs = keras.Input(shape=self.input_shape)
        scaled_input = Conv2D(3, kernel_size=1)(inputs) 
        base_model_imagenet = tf.keras.applications.ResNet50V2(
                    weights='imagenet', include_top=False, input_shape=(120,120,3))
        base_model = tf.keras.applications.ResNet50V2(
                    weights=None, include_top=False, input_tensor=scaled_input)

        for i, layer in enumerate(base_model_imagenet.layers):
            # we must skip input layer, which has no weights
            if i == 0:
                continue
            base_model.layers[i+1].set_weights(layer.get_weights())

        x = Flatten()(base_model.output)
        output = Dense(self.classes)(x)
        
        model = keras.Model(inputs=base_model.input, 
                outputs=[output, base_model.get_layer('conv2_block3_2_relu').output])

        self.model = model
