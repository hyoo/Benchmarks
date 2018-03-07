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

class UNET(default_utils.Benchmark):
    def parse_from_benchmark(self):
        pass

    def read_config_file(self, file):
        config=configparser.ConfigParser()
        config.read(file)
        section=config.sections()

        fileParams={}

        # parse the remaining values
        for k,v in config.items(section[0]):
            if not k in fileParams:
                fileParams[k] = eval(v)
        
        return fileParams

def load_data(gParameters):
    print("loading train data")
    imgs_train = np.load(gParameters['train_data'])
    imgs_train_mask = np.load(gParameters['train_label_data'])
    imgs_test = np.load(gParameters['test_data'])
    print("finish loading data")

    return imgs_train, imgs_train_mask, imgs_test 