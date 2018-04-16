from __future__ import print_function

import os
import sys
import logging
import argparse
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

import pandas as pd
import numpy as np

file_path = os.path.dirname(os.path.realpath(__file__))
lib_path = os.path.abspath(os.path.join(file_path, '..', 'common'))
sys.path.append(lib_path)
lib_path2 = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
sys.path.append(lib_path2)

import p1_common

logger = logging.getLogger(__name__)


# import sys
# file_path = os.path.dirname(os.path.realpath("__file__"))
# lib_path = os.path.abspath(os.path.join(file_path, '..', '..', 'common'))
# sys.path.append(lib_path)
import default_utils
from keras import backend as K

if K.backend() == 'tensorflow' and 'NUM_INTRA_THREADS' in os.environ:
    import tensorflow as tf
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

    # Uncommit when running on an optimized tensorflow where NUM_INTER_THREADS and
    # NUM_INTRA_THREADS env vars are set.
    print('NUM_INTER_THREADS: ', os.environ['NUM_INTER_THREADS'])
    print('NUM_INTRA_THREADS: ', os.environ['NUM_INTRA_THREADS'])
    session_conf = tf.ConfigProto(inter_op_parallelism_threads=int(os.environ['NUM_INTER_THREADS']),
          intra_op_parallelism_threads=int(os.environ['NUM_INTRA_THREADS']))
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)

from default_utils import str2bool

additional_definitions = [
    {'name':'cell_features', 'nargs':'+', 'choices':['expression', 'mirna', 'proteome', 'all', 'expression_5platform', 'expression_u133p2', 'rnaseq', 'categorical'], 'help':"use one or more cell line feature sets: 'expression', 'mirna', 'proteome', 'all'; use all for ['expression', 'mirna', 'proteome']; use 'categorical' for one-hot encoded cell lines"},
    {'name':'drug_features', 'nargs':'+', 'choices':['descriptors', 'latent', 'all', 'categorical', 'noise'], 'help':"use dragon7 descriptors, latent representations from Aspuru-Guzik's SMILES autoencoder, or both, or one-hot encoded drugs, or random features; 'descriptors','latent', 'all', 'categorical', 'noise'"},
    {'name':'dense_feature_layers', 'nargs':'+', 'type':int, 'help':'number of neurons in intermediate dense layers in the feature encoding submodels'},
    {'name':'use_landmark_genes', 'type':str2bool, 'default':'False', 'help':'use the 978 landmark genes from LINCS (L1000) as expression features'},
    {'name':'residual', 'type':str2bool, 'default':'False', 'help':'add skip connections to the layers'},
    {'name':'reduce_lr', 'type':str2bool, 'default':'False', 'help':'reduce learning rate on plateau'},
    {'name':'warmup_lr', 'type':str2bool, 'default':'False', 'help':'gradually increase learning rate on start'},
    {'name':'base_lr', 'type':float, 'default':None, 'help':'base learning rate'},
    {'name':'cp', 'type':str2bool, 'default':'False', 'help':'checkpoint models with best val_loss'},
    {'name':'tb', 'type':str2bool, 'default':'False', 'help':'use tensorboard'},
    {'name':'max_val_loss', 'type':float, 'help':'retrain if val_loss is greater than the threshold'},
    {'name':'cv_partition', 'choices':['overlapping', 'disjoint', 'disjoint_cells'], 'help':'cross validation paritioning scheme: overlapping or disjoint'},
    {'name':'cv', 'type':int, 'type':int, 'help':'cross validation folds'},
    {'name':'gen', 'type':str2bool, 'help':'use generator for training and validation data'},
    {'name':'exclude_cells', 'nargs':'+', 'default':[], 'help':'cell line IDs to exclude'},
    {'name':'exclude_drugs', 'nargs':'+', 'default':[], 'help':'drug line IDs to exclude'}
]

class Combo(default_utils.Benchmark):
    def set_locals(self):
        self.additional_definitions = additional_definitions
