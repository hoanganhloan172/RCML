#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial

# Third party imports
import tensorflow as tf
import numpy as np
import pickle


BAND_STATS = {
    'mean': {
        'B01': 340.76769064,
        'B02': 429.9430203,
        'B03': 614.21682446,
        'B04': 590.23569706,
        'B05': 950.68368468,
        'B06': 1792.46290469,
        'B07': 2075.46795189,
        'B08': 2218.94553375,
        'B8A': 2266.46036911,
        'B09': 2246.0605464,
        'B11': 1594.42694882,
        'B12': 1009.32729131
    },
    'std': {
        'B01': 554.81258967,
        'B02': 572.41639287,
        'B03': 582.87945694,
        'B04': 675.88746967,
        'B05': 729.89827633,
        'B06': 1096.01480586,
        'B07': 1273.45393088,
        'B08': 1365.45589904,
        'B8A': 1356.13789355,
        'B09': 1302.3292881,
        'B11': 1079.19066363,
        'B12': 818.86747235
    }
}


def _get_fixed_feature(size):
    """
    Creates a feature with fixed length for the bands.

    Args:
        size (int): the size of the band, e.g. 60 for 60 x 60.

    Returns:
        tf.io.FixedLenFeature: the desired feature.
    """

    return tf.io.FixedLenFeature([size * size], tf.int64)


def _get_feature_description(num_classes):
    """
    Creates a feature description for the images for later parsing.

    Args:
        num_classes (int): the number of classes in the dataset to initialize the multi hot vectors properly.

    Returns:
        dict(str -> tf.io.Feature): the created features mapped by their names.
    """

    return {
        'B01': _get_fixed_feature(20),
        'B02': _get_fixed_feature(120),
        'B03': _get_fixed_feature(120),
        'B04': _get_fixed_feature(120),
        'B05': _get_fixed_feature(60),
        'B06': _get_fixed_feature(60),
        'B07': _get_fixed_feature(60),
        'B08': _get_fixed_feature(120),
        'B8A': _get_fixed_feature(60),
        'B09': _get_fixed_feature(20),
        'B11': _get_fixed_feature(60),
        'B12': _get_fixed_feature(60),
        'BigEarthNet-19_labels_multi_hot': tf.io.FixedLenFeature([num_classes], tf.int64),
        'patch_name': tf.io.VarLenFeature(dtype=tf.string)
    }


def _normalize_band(value, key):
    """
    Normalizes a band with its specific mean and std.

    Args:
        value (tf.Tensor): The band as it has been loaded.
        key (str): The name of the band to look up the statistics.

    Returns:
        tf.Tensor: The normalized values.
    """

    return (tf.cast(value, tf.float32) - BAND_STATS['mean'][key]) / BAND_STATS['std'][key]


def _get_band(feature, name, size):
    """
    Gets a band normalized and correctly scaled from the raw data.

    Args:
        feature (obj): the feature as it was read from the files.
        name (str): the name of the band.
        size (int): the size of the band.

    Returns:
        tf.Tensor: the band parsed into a tensor for further manipulation.
    """

    return _normalize_band(tf.reshape(feature[name], [size, size]), name)


def _add_noise(noise_percentage, noisy_indices, noisy_labels_per_sample, dataisBEN, x, y, index):

    def _noisify(noise_percentage, noisy_indices, noisy_labels_per_sample, label, index):
        if index in noisy_indices:
            label[noisy_labels_per_sample[index]] = label[noisy_labels_per_sample[index]] * (-1.) + 1.
        return label, index
    
    if dataisBEN:
        label, index = tf.numpy_function(_noisify, [noise_percentage, noisy_indices, noisy_labels_per_sample, y['labels'], index], [tf.float32, tf.int64])
        return x, {'labels': label}, index 
    else:
        label, index = tf.numpy_function(_noisify, [noise_percentage, noisy_indices, noisy_labels_per_sample, y, index], [tf.int64, tf.int64])
        return x, label, index 


def _get_label(parsed_features):
    '''
    Adds noise to the label according to given indices
    '''

    return tf.cast(parsed_features['BigEarthNet-19_labels_multi_hot'], tf.float32)


def _transform_example_into_data(parsed_features):
    """
    Transforms the parsed features into tensors.

    Args:
        parsed_features (obj): the examples parsed from file.

    Returns:
        tuple: the x and y portion of the data.
    """

    return (
        {
            'B01': _get_band(parsed_features, 'B01', 20),
            'B02': _get_band(parsed_features, 'B02', 120),
            'B03': _get_band(parsed_features, 'B03', 120),
            'B04': _get_band(parsed_features, 'B04', 120),
            'B05': _get_band(parsed_features, 'B05', 60),
            'B06': _get_band(parsed_features, 'B06', 60),
            'B07': _get_band(parsed_features, 'B07', 60),
            'B08': _get_band(parsed_features, 'B08', 120),
            'B8A': _get_band(parsed_features, 'B8A', 60),
            'B09': _get_band(parsed_features, 'B09', 20),
            'B11': _get_band(parsed_features, 'B11', 60),
            'B12': _get_band(parsed_features, 'B12', 60),
            'patch_name': tf.sparse.to_dense(parsed_features['patch_name'])
        },
        {'labels': _get_label(parsed_features)}
    )


def load_archive(filenames, num_classes, batch_size=0, shuffle_size=0, test=0, noise_percentage=0., noisy_sample_indices=None, noisy_labels_per_sample=None, num_parallel_calls=10):
    """
    Loads the archive, preprocesses it as needed, and provides it as a batched, prefetched, and shuffled dataset.

    Args:
        filenames (list[str]): the TFRecord filenames to load.
        num_classes (int): the number of classes in this dataset.
        batch_size (int): the size of the batches that will be provided in this dataset. Disabled by setting it to 0.
        shuffle_size (int): the size of the shuffle buffer used when smapling batches. Disabled by setting it to 0.
        num_parallel_calls (int): the number of parallel calls, when loading the data from file.
        prefetch_size (int): the number of elements to prefetch for better throughput.
    """
    def parse_example(example):
        return _transform_example_into_data(tf.io.parse_single_example(example, feature_description))

    # Get the feature description to parse the raw data.
    feature_description = _get_feature_description(num_classes)

    # Load the TFRecord data from file.
    dataset = tf.data.TFRecordDataset(filenames)
    
    # Test the model using only a small portion of the datasets.
    if test:
        dataset = dataset.take(256)

    # Parse all entries.
    dataset = dataset.map(parse_example, num_parallel_calls)

    # Add indices per sample
    dataset_indices = tf.data.Dataset.range(0, dataset.reduce(np.int64(0), lambda x, _: x + 1))
    dataset = tf.data.Dataset.zip((dataset, dataset_indices))
    
    dataset = dataset.map(lambda xy, index: (xy[0], xy[1], index))

    # Exclude the samples that have no labels
    dataset = dataset.filter(lambda x, y, index : tf.math.not_equal(tf.math.reduce_sum(y['labels']),0))

    # Add noise to only the training dataset
    if noise_percentage != 0.:
        fn_add_noise = partial(_add_noise, noise_percentage, noisy_sample_indices, noisy_labels_per_sample, True)
        dataset = dataset.map(fn_add_noise, num_parallel_calls=num_parallel_calls)
    
    unshuffled_dataset_labels = dataset.map(lambda x, y, indices: (y))
    
    # Shuffle the data as the very first step if desired.
    if shuffle_size > 0:
        dataset = dataset.shuffle(shuffle_size, reshuffle_each_iteration=True)

    # Create a batched dataset.
    if batch_size > 0:
        dataset = dataset.batch(batch_size)

    # Prefetch some of the data to ensure maximum throughput.
    dataset = dataset.prefetch(5)

    return dataset, unshuffled_dataset_labels


def useTfData(x, y, indices, batch_size, test, noise_percentage, noisy_sample_indices, noisy_labels_per_sample):

    dataset = tf.data.Dataset.from_tensor_slices((x, y, indices))
    dataset = dataset.map(lambda x, y, indices: (x/255, y, indices))

    # Add noise to only the training dataset
    if noise_percentage != 0.:
        fn_add_noise = partial(_add_noise, noise_percentage, noisy_sample_indices, noisy_labels_per_sample, False)
        dataset = dataset.map(fn_add_noise, num_parallel_calls=10)
    
    unshuffled_dataset_labels = dataset.map(lambda x, y, indices: (y))

    # Shuffle the data as the very first step if desired.
    SHUFFLE_BUFFER_SIZE = 1000
    if SHUFFLE_BUFFER_SIZE > 0:
        dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE, reshuffle_each_iteration=True).batch(batch_size)

    # Test the model using only a small portion of the datasets
    if test:
        print('TESTING IS ON: ONLY A PORTION OF DATA IS USED.')
        dataset = dataset.take(256)

    dataset = dataset.prefetch(5)
    
    return dataset, unshuffled_dataset_labels


def load_ucmerced_set(data_path, batch_size, noise_rate, noisy_sample_indices=None, noisy_labels_per_sample=None, test=0):
    '''
    Helper function to load UC Merced Land Use data set
    '''
    sets_ = []
    train_unshuffled_dataset_labels = None
    for set_ in ['train', 'validation', 'test']:
        x = pickle.load(open(f'{data_path}/x_{set_}.pickle', 'rb'))
        y = pickle.load(open(f'{data_path}/y_{set_}.pickle', 'rb'))
        indices = pickle.load(open(f'{data_path}/{set_}_indices.pickle', 'rb'))
        dataset, unshuffled_dataset_labels = useTfData(x, y, indices, batch_size, test, noise_rate, noisy_sample_indices, noisy_labels_per_sample)
        sets_.append(dataset)
        if set_ == 'train':
            train_unshuffled_dataset_labels = unshuffled_dataset_labels

    return sets_[0], sets_[1], sets_[2], train_unshuffled_dataset_labels 
