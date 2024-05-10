'''
File: printers.py
Author: Ahmet Kerem Aksoy (a.aksoy@campus.tu-berlin.de)
'''

# Third party imports 
import tensorflow as tf

def train_printer(epoch, L2, L3, trainMetrics):
    print(f'L2 to be maximized: {L2}')
    print(f'L3 to be minimized: {L3}')
    for metric in trainMetrics.print_list:
        try:
            print(f'{metric.name} ==> {metric.result():.3f}')
        except:
            print(f'{metric.name} ==> {metric.result()}')
    print(f'Training Average Precision over epoch {epoch}: {trainMetrics.aaps.result().mean()}')

def val_printer(epoch, valMetrics):
    for metric in valMetrics.print_list:
        try:
            print(f'{metric.name} ==> {metric.result():.3f}')
        except:
            print(f'{metric.name} ==> {metric.result()}')
    print(f'Validation Average Precision over epoch {epoch}: {valMetrics.aaps.result().mean()}')
    try:
        f1 = 2.0 * (valMetrics.precision.result() * valMetrics.recall.result()) / \
            (valMetrics.precision.result() + valMetrics.recall.result())
    except:
        f1 = 0.0
    print(f'Validation F1 over epoch {epoch}: {f1:.3f}')

    classwise_precision = tf.map_fn(lambda t: t[1,1] / (t[1,1] + t[0,1]), valMetrics.confmat.result())
    print(f'Classwise Precision over epoch {epoch}: {classwise_precision}')
    classwise_recall = tf.map_fn(lambda t: t[1,1] / (t[1,1] + t[1,0]), valMetrics.confmat.result())
    print(f'Classwise Recall over epoch {epoch}: {classwise_recall}')
    try:
        classwise_f1 = (2.0 * (classwise_recall * classwise_precision)) / \
            (classwise_recall + classwise_precision)
    except:
        classwise_f1 = 0.0
    print(f'Classwise F1 over epoch {epoch}: {classwise_f1}')

    classwise_positive_labels = tf.map_fn(lambda t: t[1,0] + t[1,1], valMetrics.confmat.result())
    print(f'Classwise Positive Labels over epoch {epoch}: {classwise_positive_labels}')
    classwise_negative_labels = tf.map_fn(lambda t: t[0,1] + t[0,0], valMetrics.confmat.result())
    print(f'Classwise Negative Labels over epoch {epoch}: {classwise_negative_labels}')

def test_printer(testMetrics):
    for metric in testMetrics.print_list:
        try:
            print(f'{metric.name} ==> {metric.result():.3f}')
        except:
            print(f'{metric.name} ==> {metric.result()}')
    print(f'Test Average Precision: {testMetrics.aaps.result().mean()}')
    try:
        f1 = 2.0 * (testMetrics.precision.result() * testMetrics.recall.result()) / \
            (testMetrics.precision.result() + testMetrics.recall.result())
    except:
        f1 = 0.0
    print(f'Test F1: {f1:.3f}')

    classwise_precision = tf.map_fn(lambda t: t[1,1] / (t[1,1] + t[0,1]), testMetrics.confmat.result())
    print(f'Classwise Precision: {classwise_precision}')
    classwise_recall = tf.map_fn(lambda t: t[1,1] / (t[1,1] + t[1,0]), testMetrics.confmat.result())
    print(f'Classwise Recall: {classwise_recall}')
    try:
        classwise_f1 = (2.0 * (classwise_recall * classwise_precision)) / \
            (classwise_recall + classwise_precision)
    except:
        classwise_f1 = 0.0
    print(f'Classwise F1: {classwise_f1}')
    classwise_positive_labels = tf.map_fn(lambda t: t[1,0] + t[1,1], testMetrics.confmat.result())
    print(f'Classwise Positive Labels: {classwise_positive_labels}')
    classwise_negative_labels = tf.map_fn(lambda t: t[0,1] + t[0,0], testMetrics.confmat.result())
    print(f'Classwise Negative Labels: {classwise_negative_labels}')
