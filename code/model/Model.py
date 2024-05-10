'''
File: model.py 
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de)
'''

# Standard library imports
import os
from abc import ABC, abstractmethod

class Model(ABC):
   
    def __init__(self, name, classes, batch_size, epochs, input_shape, save_dir):
        self.model_name = f'{name}.h5'
        self.classes = classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_dir = save_dir
        self.input_shape = input_shape
        self.model = None

    def __repr__(self):
        return f'Model(self.model_name)'

    def __str__(self):
        return self.model_name

    @abstractmethod
    def build(self):
        pass

    def eval(self, X_TEST, Y_TEST):
        if self.model == None:
            raise Exception('The model is not built, call buildModel() method first')
            exit()
        
        val_loss, val_acc = self.model.evaluate(X_TEST, Y_TEST, batch_size=self.batch_size, verbose=1)
        print(f'val_loss is {val_loss} and val_acc is {val_acc}')    

    def saveModel(self, path_appendix, noise_rate):
        if self.model == None:
            raise Exception('The model is not built, call buildModel() method first')
            exit()

        # Save model and weights
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)
        if path_appendix == 'best':
            model_path = os.path.join(self.save_dir, f'{path_appendix}.h5')
        else:
            model_path = os.path.join(self.save_dir, f'{path_appendix}_{self.model_name}')
        self.model.save(model_path)
        print('Saved trained model at %s ' % model_path)

    def _get_disparity_layer_outputs(self, model, last_disparity_layer_name):
        disparity_layers = []
        for layer in model.layers:
            disparity_layers.append(layer.output)
            if layer.name == last_disparity_layer_name:
                break
        return disparity_layers

    def _get_layer_outputs(self, model, layers):
        layer_outputs = []
        for layer in layers:
            layer_outputs.append(model.get_layer(layer).output)
        return layer_outputs

    def _get_resnet50_disparity_layers(self):
        out_layers = ['conv2_block1_out','conv2_block2_out','conv2_block3_out',
                      'conv3_block1_out','conv3_block2_out','conv3_block3_out','conv3_block4_out']
        return out_layers

    def _initialize_weights(self, model, seed, initializer):
        if initializer == 'glorot':
            initializer_ = tf.keras.initializers.GlorotUniform(seed=seed)
        elif initializer == 'he':
            initializer_ = tf.keras.initializers.HeNormal(seed=seed)
        else:
            raise ValueError('Initializer could not found.')

        for layer in model.model.layers:
            layer_new_weights = []
            for layer_weights in layer.get_weights():
                weights = initializer_(np.shape(layer_weights))
                layer_new_weights.append(weights)
            layer.set_weights(layer_new_weights)
        # output_bias = tf.keras.initializers.Constant(np.log(1/model1.classes))

