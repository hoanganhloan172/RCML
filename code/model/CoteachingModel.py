from .Model import Model

import numpy as np
import tensorflow as tf    
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, Activation, LeakyReLU

class CoteachingModel(Model):
    def __init__(self, name, classes, batch_size, epochs, input_shape, save_dir):
        super().__init__(name, classes, batch_size, epochs, input_shape, save_dir)


    def build(self, dropout_rate=0.25):
        inputs = keras.Input(shape=self.input_shape)
        
        x = Conv2D(128, kernel_size=3, stride=1, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)

        x = Conv2D(128, kernel_size=3, stride=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)

        x = Conv2D(128, kernel_size=3, stride=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)

        x = MaxPooling2D(pool_size=2, strides=2)(x)
        x = Dropout(dropout_rate)(x)

        x = Conv2D(256, kernel_size=3, stride=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)

        x = Conv2D(256, kernel_size=3, stride=1, padding='same', name='l2-layer')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)
        l2_logits = x

        x = Conv2D(256, kernel_size=3, stride=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)

        x = MaxPooling2D(pool_size=2, strides=2)(x)
        x = Dropout(dropout_rate)(x)

        x = Conv2D(512, kernel_size=3, stride=1, padding='valid')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)

        x = Conv2D(256, kernel_size=3, stride=1, padding='valid')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)

        x = Conv2D(128, kernel_size=3, stride=1, padding='valid')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.01)(x)

        x = AveragePooling2D(pool_size=2)(x)
        
        x = Flatten()(x)
        x = Dense(128)(x)
        outputs = Dense(self.classes)(x)

        model = keras.Model(inputs=inputs, outputs=[outputs, l2_logits], name='coteaching_model')
    
        self.model = model
