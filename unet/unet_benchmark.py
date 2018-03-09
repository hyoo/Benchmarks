import os
import sys
file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
sys.path.append(lib_path)

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

import default_utils
import numpy as np

additional_definitions = None
required = ['batch_size', 'epochs']

class UNET(default_utils.Benchmark):
    def set_locals(self):
    if required is not None:
        self.required = set(required)
    if additional_definitions is not None:
        self.additional_definitions = additional_definitions


def load_data(gParameters):
    print("loading train data")
    imgs_train = np.load(gParameters['train_data'])
    imgs_train_mask = np.load(gParameters['train_label_data'])
    imgs_test = np.load(gParameters['test_data'])
    print("finish loading data")

    return imgs_train, imgs_train_mask, imgs_test 