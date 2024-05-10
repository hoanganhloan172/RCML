'''
File: train.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de)
'''

# Third party imports
import tensorflow as tf
import numpy as np

# Local imports
from loss_fun import loss_fun
from utils.logging.printers import train_printer
from model.model_parameters import calculate_distance_between_network_parameters
from data_prep.stackBands import prepare_input
from utils.logging.noise_comparison import compare_noisy_samples_with_potentially_noisy_samples, save_noisy_samples


def train(args,
          optimizers, 
          metrics, 
          epoch,
          swap_rate,
          logname):

    print(f'------- EPOCH {epoch} TRAINING LOSSES -------')

    samplewise_losses_1 = []
    samplewise_error_losses_1 = []
    potentially_noisy_samples_perepoch_1 = []

    samplewise_losses_2 = []
    samplewise_error_losses_2 = []
    potentially_noisy_samples_perepoch_2 = []
    
    for step, batch in enumerate(args.train_data):
        if args.label_type == 'ucmerced':
            # UC Merced is RGB default.
            x_batch_train = batch[0]
            x_batch_train = tf.cast(x_batch_train, tf.float32)
            y_batch_train = batch[1]
            y_batch_train = tf.cast(y_batch_train, tf.float32)
            batch_indices = batch[2]
        else:
            if args.channels == 'RGB':
                # Use only RGB bands
                x_batch_train = tf.stack([batch[0]['B04'], batch[0]['B03'], batch[0]['B02']], axis=3)
            else:
                x_batch_train = prepare_input(batch[0])
            y_batch_train = batch[1]['labels']
            batch_indices = batch[2]
        
        logits, losses, L3, L2, losses_to_plot1, losses_to_plot2, \
        potentially_noisy_samples = train_loop(optimizers, 
                            epoch, 
                            x_batch_train, 
                            y_batch_train, 
                            batch_indices,
                            batch, 
                            swap_rate,
                            logname,
                            {'samplewise_losses_1': samplewise_losses_1,
                             'samplewise_losses_2': samplewise_losses_2,
                             'samplewise_error_losses_1': samplewise_error_losses_1,
                             'samplewise_error_losses_2': samplewise_error_losses_2
                            },
                            args)

        probabilities_1 = tf.math.sigmoid(logits['logits_1'])
        predictions_1 = tf.cast(probabilities_1 >= args.prediction_threshold, tf.float32)
        metrics['trainMetrics_1'].update_states(y_batch_train, losses['loss_1'], predictions_1)
        potentially_noisy_samples_perepoch_1.append(potentially_noisy_samples['potentially_noisy_samples_1'])
        if not args.base:
            probabilities_2 = tf.math.sigmoid(logits['logits_2'])
            predictions_2 = tf.cast(probabilities_2 >= args.prediction_threshold, tf.float32)
            metrics['trainMetrics_2'].update_states(y_batch_train, losses['loss_2'], predictions_2)
            potentially_noisy_samples_perepoch_2.append(potentially_noisy_samples['potentially_noisy_samples_2'])

    print(f'------- EPOCH {epoch} TRAINING -------')

    train_printer(epoch, L2, L3, metrics['trainMetrics_1'])
    metrics['trainMetrics_1'].write_summary(epoch, 
                               L2, 
                               L3, 
                               losses_to_plot1)
    metrics['trainMetrics_1'].reset_states()
    save_noisy_samples(potentially_noisy_samples_perepoch_1, epoch, logname, '1')

    samplewise_losses = {
            'samplewise_losses_1': np.concatenate(samplewise_losses_1).ravel(),
            'samplewise_error_losses_1': np.concatenate(samplewise_error_losses_1).ravel()
            }

    if not args.base:
        train_printer(epoch, L2, L3, metrics['trainMetrics_2'])
        param_distances = calculate_distance_between_network_parameters(args.model1.model, args.model2.model)
        metrics['trainMetrics_2'].write_summary(epoch, 
                                   L2, 
                                   L3, 
                                   losses_to_plot2, 
                                   param_distances)
        metrics['trainMetrics_2'].reset_states()
        save_noisy_samples(potentially_noisy_samples_perepoch_2, epoch, logname, '2')

        samplewise_losses['samplewise_losses_2'] = np.concatenate(samplewise_losses_2).ravel()
        samplewise_losses['samplewise_error_losses_2'] = np.concatenate(samplewise_error_losses_2).ravel()

    print(f'-----------------------------')

    return samplewise_losses
    

def train_loop(optimizers,
               epoch, 
               x_batch_train, 
               y_batch_train,
               batch_indices,
               batch, 
               swap_rate,
               logname,
               samplewise_losses,
               args):    
    
    with tf.GradientTape(persistent=True) as tape:
        
        logits_1, l2_logits_m1 = args.model1.model(x_batch_train, training=True)
        logits = {'logits_1': logits_1, 'l2_logits_m1': l2_logits_m1}
        if not args.base:
            logits_2, l2_logits_m2 = args.model2.model(x_batch_train, training=True)
            logits['logits_2'] = logits_2
            logits['l2_logits_m2'] = l2_logits_m2

        losses, L3, L2, losses_to_plot1, losses_to_plot2, \
        potentially_noisy_samples, loss_arrays = loss_fun(y_batch_train,
                                          batch_indices,
                                          logits, 
                                          epoch,
                                          args.batch_size, 
                                          swap_rate, 
                                          args)

    grads_1 = tape.gradient(losses['loss_1'], args.model1.model.trainable_variables)
    optimizers['optimizer_1'].apply_gradients(zip(grads_1, args.model1.model.trainable_variables))
    samplewise_losses['samplewise_losses_1'].append(loss_arrays['loss_array_1'])
    samplewise_losses['samplewise_error_losses_1'].append(loss_arrays['error_loss_array_1'])

    if not args.base:
        grads_2 = tape.gradient(losses['loss_2'], args.model2.model.trainable_variables)
        optimizers['optimizer_2'].apply_gradients(zip(grads_2, args.model2.model.trainable_variables))
        samplewise_losses['samplewise_losses_2'].append(loss_arrays['loss_array_2'])
        samplewise_losses['samplewise_error_losses_2'].append(loss_arrays['error_loss_array_2'])
    
    return logits, losses, L3, L2, losses_to_plot1, losses_to_plot2, potentially_noisy_samples

