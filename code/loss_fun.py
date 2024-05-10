'''
File: loss_fun.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de)
'''

# Third party imports
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Local imports
from losses.groupLasso import groupLasso
from losses.losses import calculate_loss
from losses.l2l3_loss import l2l3_loss


def loss_fun(y_batch_train,
             batch_indices,
             logits, 
             epoch,
             batch_size, 
             swap_rate, 
             args):

    probabilities_1 = tf.math.sigmoid(logits['logits_1'])
    error_loss_array_1, classes_1 = groupLasso(y_batch_train, probabilities_1, args.miss_alpha, args.extra_beta)
    error_loss_array_1 = tf.stop_gradient(error_loss_array_1)

    if not args.base:
        probabilities_2 = tf.math.sigmoid(logits['logits_2'])
        error_loss_array_2, classes_2 = groupLasso(y_batch_train, probabilities_2, args.miss_alpha, args.extra_beta)
        error_loss_array_2 = tf.stop_gradient(error_loss_array_2)
    
    loss_array_1 = calculate_loss(args, y_batch_train, logits['logits_1'], batch_indices, epoch)
    loss_1 = tf.reduce_mean(loss_array_1)
    
    loss_arrays = {'loss_array_1': loss_array_1.numpy(), 
            'error_loss_array_1': error_loss_array_1.numpy(), 
            }

    if args.random_instead_lasso == 1:
        low2high_error_args_1 = tf.random.shuffle(tf.range(int(batch_size)))
    else:
        low2high_error_args_1 = tf.argsort(error_loss_array_1)

    low_loss_args_1 = low2high_error_args_1[:int(batch_size * swap_rate)]
    high_loss_args_1 = low2high_error_args_1[int(batch_size * swap_rate):]

    if not args.base:
        loss_array_2 = calculate_loss(args, y_batch_train, logits['logits_2'], batch_indices, epoch)

        L2, L3 = l2l3_loss(args.divergence_metric, args.lambda2, args.lambda3, 
                logits['l2_logits_m1'], logits['l2_logits_m2'], logits['logits_1'], logits['logits_2'], args.sigma)

        if args.random_instead_lasso == 1:
            low2high_error_args_2 = tf.random.shuffle(tf.range(int(batch_size)))
        else:
            low2high_error_args_2 = tf.argsort(error_loss_array_2)

        low_loss_args_2 = low2high_error_args_2[:int(batch_size * swap_rate)]
        high_loss_args_2 = low2high_error_args_2[int(batch_size * swap_rate):]
        
        if args.swap == 1:
            # Gets the low_loss_samples as conducted by the peer network
            low_loss_samples_1 = tf.gather(loss_array_1, low_loss_args_2)
            low_loss_samples_2 = tf.gather(loss_array_2, low_loss_args_1)
        elif args.swap == 0:
            # No swap 
            low_loss_samples_1 = tf.gather(loss_array_1, low_loss_args_1)
            low_loss_samples_2 = tf.gather(loss_array_2, low_loss_args_2)

        # Overwrite loss_1
        loss_1 = tf.reduce_mean(low_loss_samples_1) 

    losses_to_plot1 = {'loss_array': tf.reduce_mean(loss_array_1), 
                       'error_loss_array': tf.reduce_mean(error_loss_array_1),
                       'loss_array_75': loss_1}

    potentially_noisy_samples = {
            'potentially_noisy_samples_1': np.array(tf.gather(batch_indices, high_loss_args_1))
            }

    if not args.base:
        loss_2 = tf.reduce_mean(low_loss_samples_2)

        losses_to_plot2 = {'loss_array': tf.reduce_mean(loss_array_2), 
                           'error_loss_array': tf.reduce_mean(error_loss_array_2),
                           'loss_array_75': loss_2}
        loss_arrays['loss_array_2'] = loss_array_2.numpy()
        loss_arrays['error_loss_array_2'] = error_loss_array_2.numpy()

        potentially_noisy_samples['potentially_noisy_samples_2'] = np.array(tf.gather(batch_indices, high_loss_args_2))

        return {'loss_1': loss_1+L3-L2, 'loss_2': loss_2+L3-L2}, L3, L2, losses_to_plot1, losses_to_plot2, potentially_noisy_samples, loss_arrays

    return {'loss_1': loss_1}, 0., 0., losses_to_plot1, None, potentially_noisy_samples, loss_arrays
