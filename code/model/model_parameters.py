import tensorflow as tf

def _get_model_parameters(model):
    layers_before_mmd = []
    mmd_layer = model.outputs[1].name.split('/')[0]
    for layer in model.layers:
        layer_params = layer.trainable_weights
        if layer_params:
            layers_before_mmd.append(layer_params)
        if layer.name == mmd_layer:
            break
    network_parameters = {'parameters_before_mmd': layers_before_mmd,
                          'parameters_last_layer': [model.get_layer(index=-1).trainable_weights]
                          }

    return network_parameters


def _calculate_distance(network_params1, network_params2):
    layerwise_distances = []
    for layer_params1, layer_params2 in zip(network_params1, network_params2):
        for params1, params2 in zip(layer_params1, layer_params2):
            difference = params1.numpy() - params2.numpy()
            squared_difference = difference**2
            layerwise_distances.append(tf.math.sqrt(tf.reduce_sum(squared_difference)))
    return layerwise_distances


def _calculate_distance_between_params(params1, params2):

    layerwise_distances_before_mmd = _calculate_distance(params1['parameters_before_mmd'], 
                                                         params2['parameters_before_mmd'])
    last_layer_distance = _calculate_distance(params1['parameters_last_layer'], 
                                              params2['parameters_last_layer'])

    distance_before_mmd = tf.reduce_mean(layerwise_distances_before_mmd)
    last_layer_distance = tf.reduce_mean(last_layer_distance)

    distances = [distance_before_mmd.numpy(), 
                 last_layer_distance
                 ]

    return distances

def calculate_distance_between_network_parameters(model1, model2):
    network_parameters_1 = _get_model_parameters(model1)
    network_parameters_2 = _get_model_parameters(model2)
    params_distance_before_mmd, params_distance_last_layer = _calculate_distance_between_params(network_parameters_1,
                                      network_parameters_2)
    param_distances = {'params_distance_before_mmd': params_distance_before_mmd,
                       'params_distance_last_layer': params_distance_last_layer
                       }
    return param_distances
