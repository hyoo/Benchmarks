#! /usr/bin/env python

from __future__ import division, print_function

import numpy as np

import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats.stats import pearsonr

# import candle_keras as candle

from uno_data import CombinedDataLoader, CombinedDataGenerator

import horovod.keras as hvd
import tensorflow as tf


def discretize(y, bins=5):
    percentiles = [100 / bins * (i + 1) for i in range(bins - 1)]
    thresholds = [np.percentile(y, x) for x in percentiles]
    classes = np.digitize(y, thresholds)
    return classes


def r2(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def mae(y_true, y_pred):
    return keras.metrics.mean_absolute_error(y_true, y_pred)


def evaluate_prediction(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    return {'mse': mse, 'mae': mae, 'r2': r2, 'corr': corr}


def log_evaluation(metric_outputs, description='Comparing y_true and y_pred:'):
    print(description)
    for metric, value in metric_outputs.items():
        print('  {}: {:.4f}'.format(metric, value))


def build_feature_model(input_shape, name='', dense_layers=[1000, 1000],
                        activation='relu'):
    x_input = Input(shape=input_shape)
    h = x_input
    for i, layer in enumerate(dense_layers):
        h = Dense(layer, activation=activation)(h)

    model = Model(x_input, h, name=name)
    return model


def build_model(loader, args):
    input_models = {}
    for fea_type, shape in loader.feature_shapes.items():
        base_type = fea_type.split('.')[0]
        if base_type in ['cell', 'drug']:
            box = build_feature_model(input_shape=shape, name=fea_type,
                                      dense_layers=args.dense_feature_layers)
            input_models[fea_type] = box

    inputs = []
    encoded_inputs = []
    for fea_name, fea_type in loader.input_features.items():
        shape = loader.feature_shapes[fea_type]
        fea_input = Input(shape, name='input.'+fea_name)
        inputs.append(fea_input)
        if fea_type in input_models:
            input_model = input_models[fea_type]
            encoded = input_model(fea_input)
        else:
            encoded = fea_input
        encoded_inputs.append(encoded)

    merged = keras.layers.concatenate(encoded_inputs)

    h = merged
    for i, layer in enumerate(args.dense):
        h = Dense(layer, activation=args.activation)(h)

    output = Dense(1)(h)

    return Model(inputs, output)


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def run():
    # horovod init
    hvd.init()
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))
    print("running total:{0}, local_rank:{1}, rank:{2}".format(hvd.size(), hvd.local_rank(), hvd.rank()))

    verbose = 1 if hvd.rank() == 0 else 0
    params = {'scaling': 'std', 'tb': False, 'cell_features': ['rnaseq'], 'timeout': 3600, 'feature_subsample': 0,
              'base_lr': None, 'max_val_loss': 1.0, 'drug_median_response_min': -1, 'loss': 'mse',
              'use_landmark_genes': True, 'by_cell': None, 'rng_seed': 2018, 'run_id': 'RUN000', 'drop': 0,
              'no_gen': False, 'cp': False, 'preprocess_rnaseq': 'none', 'cache': 'cache/CTRP',
              'drug_feature_subset_path': '', 'cell_types': None, 'dense_feature_layers': [1000, 1000, 1000],
              'drug_features': ['descriptors', 'fingerprints'],
              'solr_root': '', 'drug_subset_path': '', 'validation_split': 0.2, 'batch_size': 8196,
              'export_data': None, 'logfile': None, 'no_response_source': False, 'optimizer': 'adam',
              'shuffle': False, 'single': False, 'cell_feature_subset_path': '', 'test_sources': ['train'],
              'save': 'save/uno', 'by_drug': None, 'feature_subset_path': '', 'verbose': None, 'warmup_lr': False,
              'growth_bins': 0, 'residual': False, 'dense': [1000, 1000, 1000], 'drug_median_response_max': 1,
              'agg_dose': None, 'experiment_id': 'EXP000', 'learning_rate': None, 'batch_normalization': False,
              'use_filtered_genes': False, 'train_bool': True, 'epochs': 1, 'gpus': [], 'no_feature_source': False,
              'activation': 'relu', 'cell_subset_path': '', 'train_sources': ['CTRP'], 'reduce_lr': False,
              'partition_by': None, 'cv': 1}

    args = Struct(**params)

    loader = CombinedDataLoader()
    loader.load(cache=args.cache,
                ncols=args.feature_subsample,
                agg_dose=args.agg_dose,
                cell_features=args.cell_features,
                drug_features=args.drug_features,
                drug_median_response_min=args.drug_median_response_min,
                drug_median_response_max=args.drug_median_response_max,
                use_landmark_genes=args.use_landmark_genes,
                use_filtered_genes=args.use_filtered_genes,
                cell_feature_subset_path=args.cell_feature_subset_path or args.feature_subset_path,
                drug_feature_subset_path=args.drug_feature_subset_path or args.feature_subset_path,
                preprocess_rnaseq=args.preprocess_rnaseq,
                single=args.single,
                train_sources=args.train_sources,
                test_sources=args.test_sources,
                embed_feature_source=not args.no_feature_source,
                encode_response_source=not args.no_response_source,
                )

    target = 'Growth'
    val_split = 0.2
    train_split = 1 - val_split

    loader.partition_data(cv_folds=1, train_split=train_split, val_split=val_split,
                          cell_types=args.cell_types, by_cell=args.by_cell, by_drug=args.by_drug,
                          cell_subset_path=args.cell_subset_path, drug_subset_path=args.drug_subset_path)

    model = build_model(loader, args)
    model.summary()

    # model = build_model(loader, args, silent=True)

    # optimizer = optimizers.deserialize({'class_name': args.optimizer, 'config': {}})
    # base_lr = args.base_lr or K.get_value(optimizer.lr)
    # if args.learning_rate:
    #         K.set_value(optimizer.lr, args.learning_rate * hvd.size())
    # else:
    #         K.set_value(optimizer.lr, base_lr * hvd.size())
    optimizer = keras.optimizers.Adadelta(lr=1.0 * hvd.size())
    optimizer = hvd.DistributedOptimizer(optimizer)

    # model.compile(loss=args.loss, optimizer=optimizer, metrics=[mae, r2])
    model.compile(loss=args.loss, optimizer=optimizer, metrics=['accuracy'])

    # calculate trainable and non-trainable params
    # params.update(candle.compute_trainable_params(model))

    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    ]
        # hvd.callbacks.MetricAverageCallback(),
        # hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=verbose),
        # keras.callbacks.ReduceLROnPlateau(patience=3, verbose=verbose)

    fold = 0
    train_gen = CombinedDataGenerator(loader, fold=fold, batch_size=args.batch_size, shuffle=args.shuffle,
                                      rank=hvd.rank(), total_ranks=hvd.size())
    val_gen = CombinedDataGenerator(loader, partition='val', fold=fold, batch_size=args.batch_size,
                                    shuffle=args.shuffle, rank=hvd.rank(), total_ranks=hvd.size())

    df_val = val_gen.get_response(copy=True)
    y_val = df_val[target].values
    y_shuf = np.random.permutation(y_val)
    log_evaluation(evaluate_prediction(y_val, y_shuf),
                   description='Between random pairs in y_val:')

    print('Data points per epoch: train = %d, val = %d' % (train_gen.size, val_gen.size))
    print('Steps per epoch: train = %d, val = %d' % (train_gen.steps, val_gen.steps))
    history = model.fit_generator(train_gen.flow(single=args.single), train_gen.steps,
                                  epochs=args.epochs,
                                  callbacks=callbacks,
                                  validation_data=val_gen.flow(single=args.single),
                                  validation_steps=val_gen.steps,
                                  verbose=verbose,
                                  )

    val_gen.reset()
    y_val_pred = model.predict_generator(val_gen.flow(single=args.single), val_gen.steps * 2)
    y_val_pred = y_val_pred[:val_gen.size]

    y_val_pred = y_val_pred.flatten()

    scores = evaluate_prediction(y_val, y_val_pred)
    log_evaluation(scores)

    return history


def main():
    run()


if __name__ == '__main__':
    main()
    if K.backend() == 'tensorflow':
        K.clear_session()

