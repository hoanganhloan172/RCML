'''
File: run_together.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de)
'''

# Standard library imports
from datetime import datetime
import os

# Third party imports
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Local imports
from utils.logging.loggers import TrainMetrics, ValMetrics, TestMetrics
from utils.rate_calculators.swap_rate_calculator import SwapRateCalculator
from train import train
from evaluate import evaluate
from validate import validate

def run(args):

    logdir = f'{args.logname}/scalars/'

    lr_schedule_1 = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9)
    lr_schedule_2 = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=10000,
        decay_rate=0.9)
    optimizer_1 = keras.optimizers.SGD(learning_rate=lr_schedule_1)
    optimizers = {'optimizer_1': optimizer_1}
    if not args.base:
        optimizer_2 = keras.optimizers.SGD(learning_rate=lr_schedule_2)
        optimizers['optimizer_2'] = optimizer_2

    trainMetrics_1 = TrainMetrics(logdir, '1', args.num_classes)
    valMetrics_1 = ValMetrics(logdir, '1', args.num_classes)
    testMetrics_1 = TestMetrics(logdir, '1', args.num_classes)
    metrics = {'trainMetrics_1': trainMetrics_1, 'valMetrics_1': valMetrics_1,
               'testMetrics_1': testMetrics_1}
    if not args.base:
        trainMetrics_2 = TrainMetrics(logdir, '2', args.num_classes)
        valMetrics_2 = ValMetrics(logdir, '2', args.num_classes)
        testMetrics_2 = TestMetrics(logdir, '2', args.num_classes)
        metrics['trainMetrics_2'] = trainMetrics_2
        metrics['valMetrics_2'] = valMetrics_2
        metrics['testMetrics_2'] = testMetrics_2

    swapRateCalculator = SwapRateCalculator(args.swap_start, 
            args.swap_end, args.epochs)

    samplewise_losses_1_for_n_epochs = []
    samplewise_error_losses_1_for_n_epochs = []
    if not args.base:
        samplewise_losses_2_for_n_epochs = []
        samplewise_error_losses_2_for_n_epochs = []

    for epoch in range(1,args.epochs+1):
        swap_rate = swapRateCalculator.calculate_swap_rate(epoch)
        print(f"swap_rate: {swap_rate}")
        
        # Train the model
        samplewise_losses = train(args, optimizers, 
                        metrics, epoch, swap_rate,
                        args.logname)

        samplewise_losses_1_for_n_epochs.append(samplewise_losses['samplewise_losses_1'])
        samplewise_error_losses_1_for_n_epochs.append(samplewise_losses['samplewise_error_losses_1'])
        if not args.base:
            samplewise_losses_2_for_n_epochs.append(samplewise_losses['samplewise_losses_2'])
            samplewise_error_losses_2_for_n_epochs.append(samplewise_losses['samplewise_error_losses_2'])
        
        # Validate the model
        validate(args, metrics, epoch)

    # Evaluate the last model
    evaluate(args, metrics)

    if args.logname[-2:] != '/0':
        # Load the best model
        model_path = os.path.join(args.logname, 'models/best.h5')
        best_model = tf.keras.models.load_model(model_path)
        
        # Evaluate the best model
        print('----------TEST RESULTS USING THE BEST MODEL----------')
        evaluate(args, metrics, best_model)
