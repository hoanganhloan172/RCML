'''
File: loggers.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de) 
'''

# Third party imports 
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

from .average_precision import AveragePrecision

class TrainMetrics:
    def __init__(self, logdir, name, num_classes):
        self.loss = keras.metrics.Mean(name=f'Train_Loss_{name}', dtype=tf.float32)
        self.acc = keras.metrics.Accuracy(name=f'Train_Acc_{name}')
        self.precision = keras.metrics.Precision(name=f'Train_Precision_{name}')
        self.recall = keras.metrics.Recall(name=f'Train_Recall_{name}')
        self.aaps = AveragePrecision(name=f'Per_Class_Train_Average_Precision_{name}')
        self.confmat = tfa.metrics.MultiLabelConfusionMatrix(num_classes=num_classes, name=f'Train_Confusion_Matrix_{name}')
        if logdir:
            self.summary = tf.summary.create_file_writer(logdir + f'/model_{name}/train')
        else:
            self.summary = None
        self.print_list = [self.loss, self.acc, self.precision, self.recall, self.aaps]
    
    def reset_states(self):
        self.acc.reset_states()
        self.loss.reset_states()
        self.precision.reset_states()
        self.recall.reset_states()
        self.aaps.reset_states()
        self.confmat.reset_states()
    
    def update_states(self, y_batch, loss_value, predictions):
        self.loss.update_state(loss_value)
        self.acc.update_state(y_batch, predictions)
        self.precision.update_state(y_batch, predictions)
        self.recall.update_state(y_batch, predictions)
        self.aaps.update_state(y_batch, predictions)
        self.confmat.update_state(y_batch, predictions)
    
    def write_summary(self, epoch, L2, L3, losses_to_plot, param_distances=None):
        if self.summary:
            with self.summary.as_default():
                tf.summary.scalar('Loss', self.loss.result(), step=epoch)
                tf.summary.scalar('Accuracy', self.acc.result(), step=epoch)
                tf.summary.scalar('L2', float(L2), step=epoch)
                tf.summary.scalar('L3', float(L3), step=epoch)
                tf.summary.scalar('Precision', self.precision.result(), step=epoch)
                tf.summary.scalar('Recall', self.recall.result(), step=epoch)
                f1 = 2.0 * (self.precision.result() * self.recall.result()) / \
                    (self.precision.result() + self.recall.result())
                tf.summary.scalar('F1', f1, step=epoch)
                for key, value in losses_to_plot.items():
                    tf.summary.scalar(key, value, step=epoch)
                if param_distances:
                    for key, value in param_distances.items():
                        tf.summary.scalar(key, value, step=epoch)

class ValMetrics:
    def __init__(self, logdir, name, num_classes):
        self.acc = keras.metrics.Accuracy(f'Val_Acc_{name}')
        self.precision = keras.metrics.Precision(name=f'Val_Precision_{name}')
        self.recall = keras.metrics.Recall(name=f'Val_Recall_{name}')
        self.aaps = AveragePrecision(name=f'Per_Class_Average_Precision_{name}')
        self.confmat = tfa.metrics.MultiLabelConfusionMatrix(num_classes=num_classes, name=f'Train_Confusion_Matrix_{name}')
        self.best_map_micro = 0.
        if logdir:
            self.summary = tf.summary.create_file_writer(logdir + f'/model_{name}/val')
        else:
            self.summary = None
        self.print_list = [self.acc, self.precision, self.recall, self.aaps]
    
    def reset_states(self):
        self.acc.reset_states()
        self.precision.reset_states()
        self.recall.reset_states()
        self.aaps.reset_states()
        self.confmat.reset_states()
    
    def update_states(self, y_batch, predictions):
        self.acc.update_state(y_batch, predictions)
        self.precision.update_state(y_batch, predictions)
        self.recall.update_state(y_batch, predictions)
        self.aaps.update_state(y_batch, predictions)
        self.confmat.update_state(y_batch, predictions)
    
    def write_summary(self, epoch):
        if self.summary:
            with self.summary.as_default():
                tf.summary.scalar('Accuracy', self.acc.result(), step=epoch)
                tf.summary.scalar('Precision', self.precision.result(), step=epoch)
                tf.summary.scalar('Recall', self.recall.result(), step=epoch)
                try:
                    f1 = 2.0 * (self.precision.result() * self.recall.result()) / \
                        (self.precision.result() + self.recall.result())
                except:
                    f1 = 0.0
                tf.summary.scalar('F1', f1, step=epoch)
    
    
class TestMetrics:
    def __init__(self, logdir, name, num_classes, threshold=0.35):
        self.acc = keras.metrics.Accuracy(f'Test_Acc_{name}')
        self.precision = keras.metrics.Precision(name=f'Test_Precision_{name}')
        self.recall = keras.metrics.Recall(name=f'Test_Recall_{name}')
        self.aaps = AveragePrecision(name=f'Per_Class_Test_Average_Precision_{name}')
        self.f1none = tfa.metrics.F1Score(name=f'Test_F1_None_{name}', num_classes=num_classes, threshold=threshold)
        self.f1micro = tfa.metrics.F1Score(name=f'Test_F1_Micro_{name}', num_classes=num_classes, average='micro', threshold=threshold)
        self.f1macro = tfa.metrics.F1Score(name=f'Test_F1_Macro_{name}', num_classes=num_classes, average='macro', threshold=threshold)
        self.f1weighted = tfa.metrics.F1Score(name=f'Test_F1_Weighted_{name}', num_classes=num_classes, average='weighted', threshold=threshold)
        self.confmat = tfa.metrics.MultiLabelConfusionMatrix(name=f'Test_Confusion_Matrix_{name}', num_classes=num_classes)
        if logdir:
            self.summary = tf.summary.create_file_writer(logdir + f'/model_{name}/test')
        else:
            self.summary = None
        self.print_list = [self.acc, self.precision, self.recall, self.aaps, self.f1none, 
                           self.f1micro, self.f1macro, self.f1weighted]
    
    def reset_states(self):
        self.acc.reset_states()
        self.precision.reset_states()
        self.recall.reset_states()
        self.aaps.reset_states()
        self.f1none.reset_states()
        self.f1micro.reset_states()
        self.f1macro.reset_states()
        self.f1weighted.reset_states()
        self.confmat.reset_states()
    
    def update_states(self, y_batch, predictions):
        self.acc.update_state(y_batch, predictions)
        self.precision.update_state(y_batch, predictions)
        self.recall.update_state(y_batch, predictions)
        self.aaps.update_state(y_batch, predictions)
        self.f1none.update_state(y_batch, predictions)
        self.f1micro.update_state(y_batch, predictions)
        self.f1macro.update_state(y_batch, predictions)
        self.f1weighted.update_state(y_batch, predictions)
        self.confmat.update_state(y_batch, predictions)
