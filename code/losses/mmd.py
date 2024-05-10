'''
File: mmd.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de)
'''

# Third party imports
import tensorflow as tf
import numpy as np
from sklearn import metrics

def mmd(X, Y, sigma):

    X = tf.reshape([X], [X.shape[0], -1])
    Y = tf.reshape([Y], [Y.shape[0], -1])

    XX = metrics.pairwise.rbf_kernel(X, X, 1/sigma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, 1/sigma)
    XY = metrics.pairwise.rbf_kernel(X, Y, 1/sigma)

    return XX.mean() + YY.mean() - 2 * XY.mean()

def faster_mmd(X, Y, sigma):

    X = tf.reshape([X], [X.shape[0], -1])
    Y = tf.reshape([Y], [Y.shape[0], -1])

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.linalg.tensor_diag_part(XX)
    Y_sqnorms = tf.linalg.tensor_diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)
    
    K_XX = tf.exp(-(1./sigma) * (-2. * XX + c(X_sqnorms) + r(X_sqnorms)))
    K_XY = tf.exp(-(1./sigma) * (-2. * XY + c(X_sqnorms) + r(Y_sqnorms)))
    K_YY = tf.exp(-(1./sigma) * (-2. * YY + c(Y_sqnorms) + r(Y_sqnorms)))     

    return tf.reduce_mean(K_XX) -2. * tf.reduce_mean(K_XY) + tf.reduce_mean(K_YY)
