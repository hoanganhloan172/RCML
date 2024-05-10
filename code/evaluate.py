'''
File: evaluate.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de)
'''

import os

# Third party imports 
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import average_precision_score as aps

# Local imports
from data_prep.stackBands import prepare_input
from utils.logging.printers import test_printer
from utils.logging.loggers import TestMetrics
from utils.arguments.add_arguments import add_arguments
from data_prep.prep_tf_records import load_archive, load_ucmerced_set

def main(args):

    if not args.gpu:
        tf.config.set_visible_devices([], 'GPU')

    args.logname = f'{os.getcwd()}/../output/logs/{args.logname}'

    tf.random.set_seed(0)
    np.random.seed(0)
    
    logdir = os.path.join(args.logname, 'scalars/test')
    
    # Choose which version to use
    if args.label_type == 'BEN-19':
        NUM_OF_CLASSES = 19
    elif args.label_type == 'BEN-12':
        NUM_OF_CLASSES = 12
    elif args.label_type == 'BEN-43':
        NUM_OF_CLASSES = 43
    elif args.label_type == 'ucmerced':
        NUM_OF_CLASSES = 17

    args.num_classes = NUM_OF_CLASSES 

    # Load the dataset
    if args.label_type == 'ucmerced':
        train_dataset, val_dataset, test_dataset, unshuffled_dataset_labels = load_ucmerced_set(args.dataset_path, args.batch_size, 
                                                                     args.test)
    elif args.dataset_path:  
        test_dataset, _ = load_archive(args.dataset_path + '/test.tfrecord', NUM_OF_CLASSES, 
                                    args.batch_size, 1000, args.test)
        val_dataset, _ = load_archive(args.dataset_path + '/val.tfrecord', NUM_OF_CLASSES, 
                                   args.batch_size, 1000, args.test)
    else:
        raise ValueError('Argument Error: Give the path to the folder where tf.record files are located.')
    
    # Load the best model
    model_path = os.path.join(args.logname, 'models/best.h5')
    model = tf.keras.models.load_model(model_path)
    
    testMetrics_1 = TestMetrics(logdir, '1', args.num_classes)
    metrics = {'testMetrics_1': testMetrics_1}
    if not args.base:
        testMetrics_2 = TestMetrics(logdir, '2', args.num_classes)
        metrics['testMetrics_2'] = testMetrics_2

    args.test_data = test_dataset 

    evaluate(args, 
            metrics,
            model)


def evaluate(args, metrics, best_model=None):

    model1_probabilities = []
    model2_probabilities = []
    y_true = []

    for batch_idx, batch in enumerate(args.test_data):
        
        if args.label_type == 'ucmerced':
            X = tf.cast(batch[0], tf.float32)
            Y = tf.cast(batch[1], tf.float32)
        else:
            Y = batch[1]['labels']
            if args.channels == 'RGB':
                X = tf.stack([batch[0]['B04'], batch[0]['B03'], batch[0]['B02']], axis=3)
            else:
                X = prepare_input(batch[0])
        
        if best_model:
            logits1, _ = best_model(X, training=False)
            if not args.base:
                logits2, _ = best_model(X, training=False)
        else:
            logits1, _ = args.model1.model(X, training=False)
            if not args.base:
                logits2, _ = args.model2.model(X, training=False)

        probabilities1 = tf.math.sigmoid(logits1)
        predictions1 = tf.cast(probabilities1 >= args.prediction_threshold, tf.float32)
        model1_probabilities.append(probabilities1)
        metrics['testMetrics_1'].update_states(Y, predictions1)

        if not args.base:
            probabilities2 = tf.math.sigmoid(logits2)
            predictions2 = tf.cast(probabilities2 >= args.prediction_threshold, tf.float32)
            model2_probabilities.append(probabilities2)
            metrics['testMetrics_2'].update_states(Y, predictions2)

        y_true.append(Y)

    print(f'------- TEST RESULTS -------')
    test_printer(metrics['testMetrics_1'])
    if not args.base:
        test_printer(metrics['testMetrics_2'])
    print(f'-----------------------------')

    y_true = tf.concat(y_true,0)

    print(f"y_true.shape: {y_true.shape}")
    model1_probabilities = tf.concat(model1_probabilities,0)
    print(f"model1_probabilities.shape: {model1_probabilities.shape}")
    map1_classes = aps(y_true, model1_probabilities, average=None)
    print(f"MAP over whole test set for model 1: {map1_classes}")
    print(f"MEAN MAP 1: {np.mean(map1_classes)}")
    MAP_micro_1 = aps(y_true, model1_probabilities, average='micro')
    print(f"MAP_micro_1: {MAP_micro_1}")
    MAP_macro_1 = aps(y_true, model1_probabilities, average='macro')
    print(f"MAP_macro_1: {MAP_macro_1}")

    if not args.base:
        model2_probabilities = tf.concat(model2_probabilities,0)
        print(f"model2_probabilities.shape: {model2_probabilities.shape}")
        map2_classes = aps(y_true, model2_probabilities, average=None)
        print(f"MAP over whole test set for model 2: {map2_classes}")
        print(f"MEAN MAP 2: {np.mean(map2_classes)}")
        MAP_micro_2 = aps(y_true, model2_probabilities, average='micro')
        print(f"MAP_micro_2: {MAP_micro_2}")
        MAP_macro_2 = aps(y_true, model2_probabilities, average='macro')
        print(f"MAP_macro_2: {MAP_macro_2}")

    # Save the results for the best model in a numpy array
    if best_model:
        np.save(os.path.join(args.logname, 'y_res.npy'), 
                np.array([tf.make_ndarray(tf.make_tensor_proto(y_true)), tf.make_ndarray(tf.make_tensor_proto(model1_probabilities))]),
                allow_pickle=True
                )
    
    metrics['testMetrics_1'].reset_states()
    if not args.base:
        metrics['testMetrics_2'].reset_states()


if __name__ == '__main__':
    args = add_arguments()
    main(args)
