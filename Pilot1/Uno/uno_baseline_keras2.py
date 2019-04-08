#! /usr/bin/env python

from __future__ import division, print_function

import logging
import os
import random
import sys
import time

import numpy as np
import pandas as pd

import keras
from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats.stats import pearsonr

# For non-interactive plotting
import matplotlib as mpl
import matplotlib.pyplot as plt

import uno as benchmark
import candle_keras as candle

import uno_data
from uno_data import CombinedDataLoader, CombinedDataGenerator, DataFeeder, read_feature_set_and_labels


mpl.use('Agg')
logger = logging.getLogger(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)

    random.seed(seed)

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        tf.set_random_seed(seed)
        candle.set_parallelism_threads()


def verify_path(path):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


def set_up_logger(logfile, verbose):
    verify_path(logfile)
    fh = logging.FileHandler(logfile)
    fh.setFormatter(logging.Formatter("[%(asctime)s %(process)d] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    fh.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(''))
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)

    for log in [logger, uno_data.logger]:
        log.setLevel(logging.DEBUG)
        log.addHandler(fh)
        log.addHandler(sh)


def extension_from_parameters(args):
    """Construct string for saving model with annotation of parameters"""
    ext = ''
    ext += '.A={}'.format(args.activation)
    ext += '.B={}'.format(args.batch_size)
    ext += '.E={}'.format(args.epochs)
    ext += '.O={}'.format(args.optimizer)
    # ext += '.LEN={}'.format(args.maxlen)
    ext += '.LR={}'.format(args.learning_rate)
    ext += '.CF={}'.format(''.join([x[0] for x in sorted(args.cell_features)]))
    ext += '.DF={}'.format(''.join([x[0] for x in sorted(args.drug_features)]))
    if args.feature_subsample > 0:
        ext += '.FS={}'.format(args.feature_subsample)
    if args.drop > 0:
        ext += '.DR={}'.format(args.drop)
    if args.warmup_lr:
        ext += '.wu_lr'
    if args.reduce_lr:
        ext += '.re_lr'
    if args.residual:
        ext += '.res'
    if args.use_landmark_genes:
        ext += '.L1000'
    if args.no_gen:
        ext += '.ng'
    for i, n in enumerate(args.dense):
        if n > 0:
            ext += '.D{}={}'.format(i + 1, n)
    if args.dense_feature_layers != args.dense:
        for i, n in enumerate(args.dense):
            if n > 0:
                ext += '.FD{}={}'.format(i + 1, n)

    return ext


def discretize(y, bins=5):
    percentiles = [100 / bins * (i + 1) for i in range(bins - 1)]
    thresholds = [np.percentile(y, x) for x in percentiles]
    classes = np.digitize(y, thresholds)
    return classes


def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def mae(y_true, y_pred):
    return keras.metrics.mean_absolute_error(y_true, y_pred)


def evaluate_prediction(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    corr, _ = pearsonr(y_true, y_pred)
    return {'mse': mse, 'mae': mae, 'r2': r2, 'corr': corr}


def log_evaluation(metric_outputs, description='Comparing y_true and y_pred:'):
    logger.info(description)
    for metric, value in metric_outputs.items():
        logger.info('  {}: {:.4f}'.format(metric, value))


def plot_history(out, history, metric='loss', title=None):
    title = title or 'model {}'.format(metric)
    val_metric = 'val_{}'.format(metric)
    plt.figure(figsize=(8, 6))
    plt.plot(history.history[metric], marker='o')
    plt.plot(history.history[val_metric], marker='d')
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train_{}'.format(metric), 'val_{}'.format(metric)], loc='upper center')
    png = '{}.plot.{}.png'.format(out, metric)
    plt.savefig(png, bbox_inches='tight')


class LoggingCallback(Callback):
    def __init__(self, print_fcn=print):
        Callback.__init__(self)
        self.print_fcn = print_fcn

    def on_epoch_end(self, epoch, logs={}):
        msg = "[Epoch: %i] %s" % (epoch, ", ".join("%s: %f" % (k, v) for k, v in sorted(logs.items())))
        self.print_fcn(msg)


class PermanentDropout(Dropout):
    def __init__(self, rate, **kwargs):
        super(PermanentDropout, self).__init__(rate, **kwargs)
        self.uses_learning_phase = False

    def call(self, x, mask=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(x)
            x = K.dropout(x, self.rate, noise_shape)
        return x


class ModelRecorder(Callback):
    def __init__(self, save_all_models=False):
        Callback.__init__(self)
        self.save_all_models = save_all_models
        get_custom_objects()['PermanentDropout'] = PermanentDropout

    def on_train_begin(self, logs={}):
        self.val_losses = []
        self.best_val_loss = np.Inf
        self.best_model = None

    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs.get('val_loss')
        self.val_losses.append(val_loss)
        if val_loss < self.best_val_loss:
            self.best_model = keras.models.clone_model(self.model)
            self.best_val_loss = val_loss


def build_feature_model(input_shape, name='', dense_layers=[1000, 1000],
                        activation='relu', residual=False,
                        dropout_rate=0, permanent_dropout=True):
    x_input = Input(shape=input_shape)
    h = x_input
    for i, layer in enumerate(dense_layers):
        x = h
        h = Dense(layer, activation=activation)(h)
        if dropout_rate > 0:
            if permanent_dropout:
                h = PermanentDropout(dropout_rate)(h)
            else:
                h = Dropout(dropout_rate)(h)
        if residual:
            try:
                h = keras.layers.add([h, x])
            except ValueError:
                pass
    model = Model(x_input, h, name=name)
    return model


def build_model(loader, args, permanent_dropout=True, silent=False):
    input_models = {}
    dropout_rate = args.drop
    for fea_type, shape in loader.feature_shapes.items():
        base_type = fea_type.split('.')[0]
        if base_type in ['cell', 'drug']:
            box = build_feature_model(input_shape=shape, name=fea_type,
                                      dense_layers=args.dense_feature_layers,
                                      dropout_rate=dropout_rate, permanent_dropout=permanent_dropout)
            if not silent:
                logger.debug('Feature encoding submodel for %s:', fea_type)
                box.summary(print_fn=logger.debug)
            input_models[fea_type] = box

    inputs = []
    encoded_inputs = []
    for fea_name, fea_type in loader.input_features.items():
        shape = loader.feature_shapes[fea_type]
        fea_input = Input(shape, name='input.' + fea_name)
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
        x = h
        h = Dense(layer, activation=args.activation)(h)
        if dropout_rate > 0:
            if permanent_dropout:
                h = PermanentDropout(dropout_rate)(h)
            else:
                h = Dropout(dropout_rate)(h)
        if args.residual:
            try:
                h = keras.layers.add([h, x])
            except ValueError:
                pass
    output = Dense(1)(h)

    return Model(inputs, output)


def initialize_parameters():

    # Build benchmark object
    unoBmk = benchmark.BenchmarkUno(benchmark.file_path, 'uno_default_model.txt', 'keras',
                                    prog='uno_baseline', desc='Build neural network based models to predict tumor response to single and paired drugs.')

    # Initialize parameters
    gParameters = candle.initialize_parameters(unoBmk)
    # benchmark.logger.info('Params: {}'.format(gParameters))

    return gParameters

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def run(params):
    args = Struct(**params)
    set_seed(args.rng_seed)
    ext = extension_from_parameters(args)
    verify_path(args.save)
    prefix = args.save + ext
    logfile = args.logfile if args.logfile else prefix + '.log'
    set_up_logger(logfile, args.verbose)
    logger.info('Params: {}'.format(params))

    if (len(args.gpus) > 0):
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = ",".join(map(str, args.gpus))
        K.set_session(tf.Session(config=config))

    loader = CombinedDataLoader(seed=args.rng_seed)
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

    target = args.agg_dose or 'Growth'
    val_split = args.validation_split
    train_split = 1 - val_split
        
    # 
    # Export training and validation data to HDFStore 
    #
    if args.export_data:
        export_data_fname = args.export_data

        loader.partition_data(cv_folds=args.cv, train_split=train_split, val_split=val_split,
                              cell_types=args.cell_types, by_cell=args.by_cell, by_drug=args.by_drug,
                              cell_subset_path=args.cell_subset_path, drug_subset_path=args.drug_subset_path)
        
        train_gen = CombinedDataGenerator(loader, partition='train', batch_size=args.batch_size, shuffle=args.shuffle)
        val_gen   = CombinedDataGenerator(loader, partition='val',   batch_size=args.batch_size, shuffle=args.shuffle)
            
        if os.path.exists(export_data_fname):
            os.remove(export_data_fname)

        # Store the the examples presented by the DataGenerator using pandas HDFStore
        # Each feature_set (a collection of feature colums) is stored under a distinct key
        # The 'cell_rnaseq' feature set is stored in chunks to address HDFStore column capacity
        # limitations.
        #
        # The full 'cell_rnaseq' is too wide for direct HDFS H5 representation, Break it up 
        # into gulp_size chunks and write each independently. The HDFStore column names are replaced with
        # null-strings because (1) the header storage space is limitted and (2) they are not needed.

        store = pd.HDFStore(export_data_fname, complevel=9, complib='blosc:snappy')
    
        # save cell.rnaseq feature names
        loader.cell_feature_names = loader.df_cell_rnaseq.columns[1:]    
        df_feature = pd.DataFrame(loader.cell_feature_names, columns=['rnaseq'])
        store.put('feature_names', df_feature)

        cell_rnaseq_ndx = None
        input_feature_list = [fname for fname in loader.input_features]
        if input_feature_list.count('cell.rnaseq') > 0:
            cell_rnaseq_ndx = input_feature_list.index('cell.rnaseq')

        # single/multiple drug feature sets 
        feature_itemsizes = {'Sample':30, 'Drug1':10}
        if args.single == False:
            feature_itemsizes['Drug2'] = 10

        for partition in ['train', 'val']:
            gen = train_gen if partition == 'train' else val_gen
            for i in range(gen.steps):
                x_list, y = gen.get_slice(size=args.batch_size, dataframe=True, single=args.single)
                for j, input_feature in enumerate(x_list):
                    input_feature = input_feature.astype('float32')     
                    store_key = 'x_{}_{}'.format(partition, j)                  # HDFStore key, major 
                   
                    if j == cell_rnaseq_ndx:
                        chunk_size  = 4096
                        _, nbr_cols = input_feature.shape
                        nbr_passes  = (nbr_cols + chunk_size - 1) // chunk_size
                       
                        # store feature set in chunks
                        for curr in range(nbr_passes): 
                            store_subkey = '{}_{}'.format(store_key, curr)      # HDFStore key, minor
                            colorg = chunk_size * curr
                            colfin = colorg + chunk_size    
                            subset = input_feature.iloc[:, colorg:colfin]
                            subset.columns = [''] * len(subset.columns)
                            store.append(store_subkey, subset, format='table')  # store sharded feature_set  
                        continue

                    # store standard (unchunked) feature set
                    input_feature.columns = [''] * len(input_feature.columns)
                    store.append(store_key, input_feature, format='table')

                # store labels
                store.append('y_{}'.format(partition), y.astype({target:'float32'}), format='table',
                    min_itemsize=feature_itemsizes
                )

                logger.info('Generating {} dataset. {} / {}'.format(partition, i, gen.steps))

                """
                # TEST TEST TEST - Read back and reassemble the first feature set group written above
                if i == 0:
                    x, y = read_feature_set_and_labels(
                        nbr_features = len(x_list),
                        cell_feature_ndx = cell_rnaseq_ndx,
                        loader = loader,
                        partition = partition,
                        HDFStore = store,
                        start = 0,
                        stop  = 4096
                    )    
                """
        # cleanup
        store.close()

        logger.info('Completed generating {}'.format(export_data_fname))
        return

    # If training/evaluating with previously exported data, recover the names of the individual
    # features associated with the cell_rnaseq feature set from the HDFStore file
    
    if args.use_exported_data:
        export_file = args.use_exported_data 
        if not os.path.exists(export_file):
            sys.exit("Exported data file: %s not found" % export_file)

        export_store = pd.HDFStore(export_file)
        names = export_store.get('feature_names')
        name_list = names.iloc[:,0].tolist()
        loader.cell_feature_names = {key:seq for (seq, key) in enumerate(name_list)}
        export_store.close()

    # The 'cell_feature_subset_path' argument can be specified with 'use_exported_data' to define
    # a subset of features (genes) from the complete 'cell_rnaseq' feature set. It is optional,
    # however, and all features are selected in the absence of that arg.

    if args.cell_feature_subset_path and args.use_exported_data:
        export_file = args.use_exported_data 
        feature_file = args.cell_feature_subset_path
        if not os.path.exists(feature_file):
            sys.exit("Cell feature selection file: %s not found" % feature_file)

        loader.select_cell_features(feature_file)
        loader.create_cell_features_extract_file(feature_file, export_file)  

    #
    # Build the model 
    #
    loader.partition_data(cv_folds=args.cv, train_split=train_split, val_split=val_split,
                          cell_types=args.cell_types, by_cell=args.by_cell, by_drug=args.by_drug,
                          cell_subset_path=args.cell_subset_path, drug_subset_path=args.drug_subset_path)

    model = build_model(loader, args)
    logger.info('Combined model:')
    model.summary(print_fn=logger.info)
    # plot_model(model, to_file=prefix+'.model.png', show_shapes=True)

    #
    # Save model in JSON format
    #
    if args.cp:
        model_json = model.to_json()
        with open(prefix + '.model.json', 'w') as f:
            print(model_json, file=f)

    def warmup_scheduler(epoch):
        lr = args.learning_rate or base_lr * args.batch_size / 100
        if epoch <= 5:
            K.set_value(model.optimizer.lr, (base_lr * (5 - epoch) + lr * epoch) / 5)
        logger.debug('Epoch {}: lr={:.5g}'.format(epoch, K.get_value(model.optimizer.lr)))
        return K.get_value(model.optimizer.lr)

    df_pred_list = []

    cv_ext = ''
    cv = args.cv if args.cv > 1 else 1

    for fold in range(cv):
        if args.cv > 1:
            logger.info('Cross validation fold {}/{}:'.format(fold + 1, cv))
            cv_ext = '.cv{}'.format(fold + 1)

        if len(args.gpus) > 1:
            from keras.utils import multi_gpu_model
            gpu_count = len(args.gpus)
            model = multi_gpu_model(build_model(loader, args, silent=True), cpu_merge=False, gpus=gpu_count)
        else:
            model = build_model(loader, args, silent=True)

        optimizer = optimizers.deserialize({'class_name': args.optimizer, 'config': {}})
        base_lr = args.base_lr or K.get_value(optimizer.lr)
        if args.learning_rate:
            K.set_value(optimizer.lr, args.learning_rate)

        model.compile(loss=args.loss, optimizer=optimizer, metrics=[mae, r2])

        # calculate trainable and non-trainable params
        params.update(candle.compute_trainable_params(model))

        candle_monitor = candle.CandleRemoteMonitor(params=params)
        timeout_monitor = candle.TerminateOnTimeOut(params['timeout'])

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        warmup_lr = LearningRateScheduler(warmup_scheduler)
        checkpointer = MultiGPUCheckpoint(prefix + cv_ext + '.model.h5', save_best_only=True)
        tensorboard = TensorBoard(log_dir="tb/{}{}{}".format(args.tb_prefix, ext, cv_ext))
        history_logger = LoggingCallback(logger.debug)
        model_recorder = ModelRecorder()

        # callbacks = [history_logger, model_recorder]
        # callbacks = [candle_monitor, timeout_monitor, history_logger, model_recorder]
        callbacks = []
        # callbacks = [candle_monitor, history_logger, model_recorder]  #
        if args.reduce_lr:
            callbacks.append(reduce_lr)
        if args.warmup_lr:
            callbacks.append(warmup_lr)
        if args.cp:
            callbacks.append(checkpointer)
        if args.tb:
            callbacks.append(tensorboard)

        if args.use_exported_data is not None:
            data_arg  = args.use_exported_data
            train_gen = DataFeeder(loader, partition='train', datafile=data_arg, batch_size=args.batch_size, shuffle=args.shuffle)
            val_gen = DataFeeder(loader, partition='val', datafile=data_arg, batch_size=args.batch_size, shuffle=args.shuffle)
        else:
            train_gen = CombinedDataGenerator(loader, fold=fold, batch_size=args.batch_size, shuffle=args.shuffle)
            val_gen = CombinedDataGenerator(loader, partition='val', fold=fold, batch_size=args.batch_size, shuffle=args.shuffle)

        df_val = val_gen.get_response(copy=True)
        y_val = df_val[target].values
        y_shuf = np.random.permutation(y_val)
        log_evaluation(evaluate_prediction(y_val, y_shuf),
                       description='Between random pairs in y_val:')

        print("Batch size: %d Nbr epochs: %d" % (args.batch_size, args.epochs))
        cpu_tm_org  = time.process_time()
        wall_tm_org = time.time()

        if args.no_gen:
            x_train_list, y_train = train_gen.get_slice(size=train_gen.size, single=args.single)
            x_val_list, y_val = val_gen.get_slice(size=val_gen.size, single=args.single)
            history = model.fit(x_train_list, y_train,
                                batch_size=args.batch_size,
                                epochs=args.epochs,
                                callbacks=callbacks,
                                validation_data=(x_val_list, y_val))
        else:
            logger.info('Data points per epoch: train = %d, val = %d', train_gen.size, val_gen.size)
            logger.info('Steps per epoch: train = %d, val = %d', train_gen.steps, val_gen.steps)
            history = model.fit_generator(train_gen, train_gen.steps,
                                          epochs=args.epochs,
                                          callbacks=callbacks,
                                          validation_data=val_gen,
                                          validation_steps=val_gen.steps)

        wall_tm_end = time.time()
        cpu_tm_end  = time.process_time() 
        wall_secs   = wall_tm_end - wall_tm_org
        cpu_secs    = cpu_tm_end  - cpu_tm_org

        total_samples        = train_gen.size + val_gen.size
        samples_per_wall_sec = (total_samples * args.epochs) / wall_secs
        samples_per_cpu_sec  = (total_samples * args.epochs) / cpu_secs
    
        print('Elapsed time: %.3f, CPU seconds: %.3f' % (wall_secs, cpu_secs))
        print('Samples per second (elapsed): %.3f, CPU: %.3f' % (samples_per_wall_sec, samples_per_cpu_sec))

        if args.cp:
            model = model_recorder.best_model
            model.save(prefix+'.model.h5')
            # model.load_weights(prefix+cv_ext+'.weights.h5')

        if args.no_gen:
            y_val_pred = model.predict(x_val_list, batch_size=args.batch_size)
        else:
            val_gen.reset()
            y_val_pred = model.predict_generator(val_gen, val_gen.steps + 1)
            y_val_pred = y_val_pred[:val_gen.size]

        y_val_pred = y_val_pred.flatten()

        scores = evaluate_prediction(y_val, y_val_pred)
        log_evaluation(scores)

        df_val = df_val.assign(PredictedGrowth=y_val_pred, GrowthError=y_val_pred - y_val)
        df_val['Predicted' + target] = y_val_pred
        df_val[target + 'Error'] = y_val_pred - y_val
        df_pred_list.append(df_val)

        if hasattr(history, 'loss'):
            plot_history(prefix, history, 'loss')
        if hasattr(history, 'r2'):
            plot_history(prefix, history, 'r2')

    pred_fname = prefix + '.predicted.tsv'
    df_pred = pd.concat(df_pred_list)
    if args.agg_dose:
        df_pred.sort_values(['Source', 'Sample', 'Drug1', 'Drug2', target], inplace=True)
    else:
        df_pred.sort_values(['Sample', 'Drug1', 'Drug2', 'Dose1', 'Dose2', 'Growth'], inplace=True)
    df_pred.to_csv(pred_fname, sep='\t', index=False, float_format='%.4g')

    if args.cv > 1:
        scores = evaluate_prediction(df_pred[target], df_pred['Predicted' + target])
        log_evaluation(scores, description='Combining cross validation folds:')

    for test_source in loader.test_sep_sources:
        test_gen = CombinedDataGenerator(loader, partition='test', batch_size=args.batch_size, source=test_source)
        df_test = test_gen.get_response(copy=True)
        y_test = df_test[target].values
        n_test = len(y_test)
        if n_test == 0:
            continue
        if args.no_gen:
            x_test_list, y_test = test_gen.get_slice(size=test_gen.size, single=args.single)
            y_test_pred = model.predict(x_test_list, batch_size=args.batch_size)
        else:
            y_test_pred = model.predict_generator(test_gen.flow(single=args.single), test_gen.steps)
            y_test_pred = y_test_pred[:test_gen.size]
        y_test_pred = y_test_pred.flatten()
        scores = evaluate_prediction(y_test, y_test_pred)
        log_evaluation(scores, description='Testing on data from {} ({})'.format(test_source, n_test))

    if K.backend() == 'tensorflow':
        K.clear_session()

    logger.handlers = []

    return history


def main():
    params = initialize_parameters()
    run(params)


if __name__ == '__main__':
    main()
    if K.backend() == 'tensorflow':
        K.clear_session()
