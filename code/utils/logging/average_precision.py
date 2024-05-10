from functools import partial
from sklearn.metrics import average_precision_score as aps
import numpy as np

class AveragePrecision:
    '''
    sklearn.metrics.average_precision_score in tensorflow style.
    Why? Because keeping states is a better paradigm for training things.
    '''
    def __init__(self, name):
        self.name = name
        self.aaps = partial(aps, average=None)
        self.state_result = 0.
        self.state_num_batches = 0.

    def reset_states(self):
        self.state_result = 0.
        self.state_num_batches = 0.

    def update_state(self, y_true, y_pred):
        self.state_result += np.nan_to_num(self.aaps(y_true, y_pred))
        self.state_num_batches += 1.

    def result(self):
        return self.state_result / self.state_num_batches
