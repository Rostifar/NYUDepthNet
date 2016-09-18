import numpy as np
from os import listdir
from os.path import splitext
import tensorflow as tf


def _get_parameters_in_directory(directory):
    params = {}
    for param_file in listdir(directory):
        if param_file.endswith('.npy' or '.npz'):
            loaded_param_arr = np.load(param_file)
            file_name = splitext(param_file)
            params[file_name] = (convert_to_tensorflow_kernel(loaded_param_arr))
    return params


def get_weights(weights_dir):
    weights = {}
    for network_name in listdir(weights_dir):
        weights[str(network_name)] = _get_parameters_in_directory(weights_dir + "/" + network_name)
    return weights


def convert_to_tensorflow_kernel(kernel):
    """
    :param kernel: theano weight kernel
    :return: tensorflow compatible weight kernel
    """
    width = kernel.shape[0]
    height = kernel.shape[1]

    tensorflow_kernel = np.copy(kernel)

    for var_i in range(width):
        for var_j in range(height):
            tensorflow_kernel[var_i, var_j, :, :] = kernel[width - var_i - 1, height - var_j - 1, :, :]
    return tensorflow_kernel
