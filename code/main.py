'''
File: main.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de) 
'''

# Standard library imports
import os

# Third party imports
import tensorflow as tf
import numpy as np
from tensorflow import keras
import numpy as np

# Local imports
from utils.arguments.add_arguments import add_arguments
from run import run
from model.ModelFactory import ModelFactory
from data_prep.prep_tf_records import load_archive, load_ucmerced_set
from utils.logging.setup_logger import setup_logger
from losses.losses import SelfAdaptiveTrainingCE, ELR
from data_prep.get_noisy_sample_indices import get_noisy_sample_indices, get_noisy_labels_per_noisy_sample

def main(args):

    if not args.gpu:
        tf.config.set_visible_devices([], 'GPU')

    args.logname = f'../output/logs/{args.logname}'

    # Setup loggers
    args.noise_comparison_logger = setup_logger('noise_comparison', args.logname, 'noise_camparison.log')

    # Set seeds
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    # Choose which version to use
    if args.label_type == 'BEN-19':
        NUM_OF_CLASSES = 19
        NUM_TRAINING_SAMPLES = 8197
    elif args.label_type == 'BEN-12':
        NUM_OF_CLASSES = 12
        NUM_TRAINING_SAMPLES = 8192
    elif args.label_type == 'BEN-43':
        NUM_OF_CLASSES = 43
        NUM_TRAINING_SAMPLES = 8197
    elif args.label_type == 'ucmerced':
        NUM_OF_CLASSES = 17
        NUM_TRAINING_SAMPLES = 1470
    

    NOISY_SAMPLE_INDICES = get_noisy_sample_indices(NUM_TRAINING_SAMPLES, args.sample_rate)
    if len(NOISY_SAMPLE_INDICES) != 0: 
        NOISY_LABELS_PER_SAMPLE = get_noisy_labels_per_noisy_sample(NUM_TRAINING_SAMPLES, NUM_OF_CLASSES, NOISY_SAMPLE_INDICES, args.class_rate)
        with open(os.path.join(args.logname, 'noisy_samples.npy'), 'wb') as f:
            np.save(f, np.array(NOISY_SAMPLE_INDICES))
        with open(os.path.join(args.logname, 'noisy_labels.npy'), 'wb') as f:
            np.save(f, np.array(NOISY_LABELS_PER_SAMPLE))
    else:
        NOISY_SAMPLE_INDICES = None
        NOISY_LABELS_PER_SAMPLE = None

    # Load the dataset
    if args.label_type == 'ucmerced':
        train_dataset, val_dataset, test_dataset, unshuffled_dataset_labels = load_ucmerced_set(args.dataset_path, args.batch_size, 
                args.class_rate, NOISY_SAMPLE_INDICES, NOISY_LABELS_PER_SAMPLE, args.test)
    elif args.dataset_path:  
        train_dataset, unshuffled_dataset_labels = load_archive(args.dataset_path + '/train.tfrecord', NUM_OF_CLASSES, 
                args.batch_size, 1000, args.test, args.class_rate, NOISY_SAMPLE_INDICES, NOISY_LABELS_PER_SAMPLE)
        test_dataset, _ = load_archive(args.dataset_path + '/test.tfrecord', NUM_OF_CLASSES,
                                    args.batch_size, 1000, args.test)
        val_dataset, _ = load_archive(args.dataset_path + '/val.tfrecord', NUM_OF_CLASSES, 
                                   args.batch_size, 1000, args.test)
    else:
        raise ValueError('Argument Error: Give the path to the folder where tf.record files are located.')

    modelFactory = ModelFactory(NUM_OF_CLASSES, args.batch_size, args.epochs, f'{args.logname}/models')
    model1 = modelFactory.create_model('model1', args.channels, args.architecture, args.label_type)
    model1.build()
    args.model1 = model1
    if not args.base:
        model2 = modelFactory.create_model('model2', args.channels, args.architecture, args.label_type)
        model2.build()
        args.model2 = model2
    
    args.train_data = train_dataset
    args.val_data = val_dataset
    args.test_data = test_dataset 
    args.lambda2 = args.lambda_two
    args.lambda3 = args.lambda_three
    args.num_classes = NUM_OF_CLASSES 

    # Some losses need to initialized with default values and labels
    if args.loss_fn == 'SAT':
        if args.label_type == 'ucmerced':
            all_labels = tf.cast(tf.stack([y for y in unshuffled_dataset_labels]), dtype=tf.float32)
        else:
            all_labels = tf.cast(tf.stack([y['labels'] for y in unshuffled_dataset_labels]), dtype=tf.float32)
        args.SAT = SelfAdaptiveTrainingCE(all_labels)
    elif args.loss_fn == 'ELR':
        args.ELR = ELR(NUM_TRAINING_SAMPLES, NUM_OF_CLASSES)
    elif args.loss_fn == 'JOCOR':
        args.JoCor = LossJoCoR()

    # Run
    run(args)

    # Summarize models
    args.model1.saveModel('last', args.sample_rate)  
    if not args.base:
        args.model2.saveModel('last', args.sample_rate)  

if __name__ == '__main__':
    args = add_arguments()
    main(args)
