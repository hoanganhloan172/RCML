from .Model import Model

class PaperModel(Model):
    def __init__(self, name, classes, batch_size, epochs, input_shape, save_dir):
        super().__init__(name, classes, batch_size, epochs, input_shape, save_dir)

    def build(self):
        inputs = keras.Input(shape=self.input_shape)
        x = Conv2D(128, (3,3), padding='same')(inputs)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3,3), padding='same', name='l2-layer')(x)
        x = LeakyReLU(alpha=0.01)(x)
        l2_logits = x
        x = BatchNormalization()(x) # REMOVE THIS MAYBE???
        x = Conv2D(256, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(512, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = Conv2D(256, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3,3), padding='same')(x)
        x = LeakyReLU(alpha=0.01)(x)
        x = BatchNormalization()(x)
        x = AveragePooling2D(pool_size=(2,2), strides=(2,2))(x)
        
        x = Flatten()(x)
        x = Dense(128)(x)
        outputs = Dense(self.classes)(x)

        model = keras.Model(inputs=inputs, outputs=[outputs, l2_logits], name='dct_model')
    
        self.model = model
