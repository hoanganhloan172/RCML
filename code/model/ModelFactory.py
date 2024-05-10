import tensorflow as tf
import numpy as np

from .CoteachingModel import CoteachingModel
from .PaperModel import PaperModel
from .DensenetModel import DensenetModel
from .ResnetModel import ResnetModel
from .ResnetMultispectralModel import ResnetMultispectralModel

class ModelFactory:
    def __init__(self, num_classes, batch_size, epochs, logname):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.logname = logname

    def create_model(self, name, channels, architecture, label_type): 

        model_args = [self.num_classes,
                      self.batch_size,
                      self.epochs,
                      self.derive_input_shape(channels, label_type),
                      self.logname]

        if architecture == 'resnet':
            if channels == 10:
                return ResnetMultispectralModel(name, *model_args)
            elif channels == 3:
                return ResnetModel(name, *model_args)
        elif architecture == 'densenet':
            return DensenetModel(name, *model_args)
        elif architecture == 'coteaching':
            return CoteachingModel(name, *model_args)
        elif architecture == 'papermodel':
            return PaperModel(name, *model_args)
        else:
            raise Exception('Architecture not found!')

    def derive_input_shape(self, channels, label_type):
        if label_type == 'ucmerced':
            return (256, 256, channels)
        else:
            return (120, 120, channels)
