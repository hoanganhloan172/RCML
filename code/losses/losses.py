import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np


def calculate_loss(args, y, logits, batch_indices, epoch):
    if args.loss_fn == 'FOCAL':
        loss_array = focal_loss(y, logits) 
    elif args.loss_fn == 'BCE':
        loss_array = bce_loss(y, logits)
    elif args.loss_fn == 'SAT':
        loss_array = args.SAT.loss_fn(y, logits, batch_indices, epoch)
    elif args.loss_fn == 'ELR':
        loss_array = args.ELR.loss_fn(y, logits, batch_indices)
    return loss_array


def bce_loss(y_batch_train, logits):
    loss_array = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(y_batch_train, logits, 1.0), axis=1)
    return loss_array


def focal_loss(y_batch_train, logits):
    loss_array = tfa.losses.sigmoid_focal_crossentropy(y_batch_train, logits, from_logits=True)
    return loss_array


class SelfAdaptiveTrainingCE:
    def __init__(self, labels, momentum=0.9, es=60):
        self.bce_with_logits = tf.nn.sigmoid_cross_entropy_with_logits
        self.soft_labels = tf.identity(labels)
        self.momentum = momentum
        self.es = es

    def loss_fn(self, labels, logits, indices, epoch):

        if epoch - 1 < self.es:
            return tf.reduce_mean(self.bce_with_logits(labels, logits), axis=1)

        # obtain prob, then update running avg
        prob = tf.math.sigmoid(tf.stop_gradient(logits))
        
        updates = self.momentum * tf.gather(self.soft_labels, indices) + (1. - self.momentum) * prob
        self.soft_labels = tf.tensor_scatter_nd_update(self.soft_labels, np.expand_dims(indices, 1), updates)

        # compute cross entropy loss, without reduction
        loss = tf.reduce_mean(self.bce_with_logits(tf.gather(self.soft_labels, indices), logits), axis=1)
        return loss


class ELR:
    def __init__(self, num_examp, num_classes=17, lam=1, beta=0.9):
        """Early Learning Regularization.
        Parameters
        * `num_examp` Total number of training examples.
        * `num_classes` Number of classes in the classification problem.
        * `lambda` Regularization strength; must be a positive float, controling the strength of the ELR.
        * `beta` Temporal ensembling momentum for target estimation.
        """
        self.num_classes = num_classes
        self.target = tf.zeros([num_examp, self.num_classes], dtype=tf.float32)
        self.beta = beta
        self.lam = lam
        self.bce_with_logits = tf.nn.sigmoid_cross_entropy_with_logits

    def loss_fn(self, labels, logits, indices):
        y_pred = tf.math.sigmoid(logits)
        y_pred = tf.clip_by_value(y_pred, 1e-4, 1.0 - 1e-4)
        y_pred_ = tf.stop_gradient(y_pred)

        updates = self.beta * tf.gather(self.target, indices) + (1 - self.beta) * y_pred_
        self.target = tf.tensor_scatter_nd_update(self.target, np.expand_dims(indices, 1), updates)

        ce_loss = tf.reduce_mean(self.bce_with_logits(labels, logits), axis=1)
        elr_reg = tf.reduce_mean(tf.math.log((1 - (tf.gather(self.target, indices) * y_pred))), axis=1)
        final_loss = ce_loss + self.lam * elr_reg
        return final_loss

