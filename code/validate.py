'''
File: validate.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de)
'''

# Third party imports
import tensorflow as tf
from sklearn.metrics import average_precision_score as aps

# Local imports
from data_prep.stackBands import prepare_input
from utils.logging.printers import val_printer

def validate(args, metrics, epoch, best_model=None):

    model1_probabilities = []
    model2_probabilities = []
    y_true = []

    for batch in args.val_data:
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
            logits1, logits_m1 = best_model(X, training=False)
            if not args.base:
                logits2, logits_m2 = best_model(X, training=False)
        else:
            logits1, logits_m1 = args.model1.model(X, training=False)
            if not args.base:
                logits2, logits_m2 = args.model2.model(X, training=False)

        probabilities1 = tf.math.sigmoid(logits1)
        predictions1 = tf.cast(probabilities1 >= args.prediction_threshold, tf.float32)
        model1_probabilities.append(probabilities1)
        metrics['valMetrics_1'].update_states(Y, predictions1)

        if not args.base:
            probabilities2 = tf.math.sigmoid(logits2)
            predictions2 = tf.cast(probabilities2 >= args.prediction_threshold, tf.float32)
            model2_probabilities.append(probabilities2)
            metrics['valMetrics_2'].update_states(Y, predictions2)

        y_true.append(Y)
        
    print(f'------- EPOCH {epoch} VALIDATION -------')
    val_printer(epoch, metrics['valMetrics_1'])
    metrics['valMetrics_1'].write_summary(epoch)
    if not args.base:
        val_printer(epoch, metrics['valMetrics_2'])
        metrics['valMetrics_2'].write_summary(epoch)
    print(f'-----------------------------')

    y_true = tf.concat(y_true,0)
    model1_probabilities = tf.concat(model1_probabilities,0)
    MAP_micro_1 = aps(y_true, model1_probabilities, average='micro')
    print(f"MAP_micro_1: {MAP_micro_1}")

    if not args.base:
        model2_probabilities = tf.concat(model2_probabilities,0)
        MAP_micro_2 = aps(y_true, model2_probabilities, average='micro')
        print(f"MAP_micro_2: {MAP_micro_2}")

    if not best_model:
        if not args.base:
            save_best_model(metrics, MAP_micro_1, MAP_micro_2, args, epoch)
        else:
            save_best_model(metrics, MAP_micro_1, 0., args, epoch)
     
    metrics['valMetrics_1'].reset_states()

    if not args.base:
        metrics['valMetrics_2'].reset_states()


def save_best_model(metrics, MAP_micro_1, MAP_micro_2, args, epoch):
    # Choose the better map micro score among two
    new_map_micro = MAP_micro_1 if MAP_micro_1 > MAP_micro_2 else MAP_micro_2
    
    # Choose the better map micro score's model
    new_model = args.model1 if MAP_micro_1 > MAP_micro_2 else args.model2
    
    if new_map_micro > metrics['valMetrics_1'].best_map_micro:
        # Save the model
        new_model.saveModel('best', args.sample_rate)
        print(f'New best model at epoch {epoch} with map micro score of {new_map_micro}')
        
        # Update the best map micro
        metrics['valMetrics_1'].best_map_micro = new_map_micro
        if MAP_micro_2 != 0.:
            metrics['valMetrics_2'].best_map_micro = new_map_micro
