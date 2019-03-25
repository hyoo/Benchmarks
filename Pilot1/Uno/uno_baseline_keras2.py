#! /usr/bin/env python

from __future__ import division, print_function

import argparse
import collections
import logging
import os
import random
import threading

import numpy as np
import pandas as pd

import keras
from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler, TensorBoard
from keras.utils import get_custom_objects
from keras.utils.vis_utils import plot_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from scipy.stats.stats import pearsonr

# For non-interactive plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import uno as benchmark
import candle_keras as candle

import uno_data
from uno_data import CombinedDataLoader, CombinedDataGenerator, DataFeeder


logger = logging.getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(seed)

    random.seed(seed)

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        tf.set_random_seed(seed)
        # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        # K.set_session(sess)

        # Uncommit when running on an optimized tensorflow where NUM_INTER_THREADS and
        # NUM_INTRA_THREADS env vars are set.
        # session_conf = tf.ConfigProto(inter_op_parallelism_threads=int(os.environ['NUM_INTER_THREADS']),
        #	intra_op_parallelism_threads=int(os.environ['NUM_INTRA_THREADS']))
        # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
        # K.set_session(sess)


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
            ext += '.D{}={}'.format(i+1, n)
    if args.dense_feature_layers != args.dense:
        for i, n in enumerate(args.dense):
            if n > 0:
                ext += '.FD{}={}'.format(i+1, n)

    return ext


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
    #benchmark.logger.info('Params: {}'.format(gParameters))

    return gParameters

def report_duplicates(as_list, as_set, name='list'):
    copy_set  = as_set.copy()
    sort_list = as_list[:]
    sort_list.sort()

    for mem in sort_list:
        if mem in copy_set:
            copy_set -= {mem}
        else:
            print('   member: %s appears multiple times in %s' % (mem, name)) 

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def run(params):
    args = Struct(**params)
    set_seed(args.rng_seed)
    ext = extension_from_parameters(args)
    verify_path(args.save)
    prefix = args.save + ext
    logfile = args.logfile if args.logfile else prefix+'.log'
    set_up_logger(logfile, args.verbose)
    logger.info('Params: {}'.format(params))

    if (len(args.gpus) > 0):
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = ",".join(map(str,args.gpus))
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

        # Loader extensions supporting pre-joined source data
        # cell_feature_names is a list of all feature_names (columns)  associated with feature_set == 2 - named 'cell.rnaseq'
        # That file is stored as '<--export_data>.rnaseq'

        loader.cell_feature_names = None
        cell_feature_names_container = 'rnaseq'
        cell_feature_names_filename  = '{}.{}_names'.format(export_data_fname, cell_feature_names_container) 
            
        if os.path.exists(export_data_fname):
            os.remove(export_data_fname)

        # Store the the examples presented by the DataGenerator using pandas HDFStore
        # Each feature_set (a collection of feature colums) is stored under a distinct key
        # The 'cell_rnaseq' feature set is stored in chunks to address HDFStore column capacity limitations

        store = pd.HDFStore(export_data_fname, complevel=9, complib='blosc:snappy')

        for partition in ['train', 'val']:
            gen = train_gen if partition == 'train' else val_gen
            for i in range(gen.steps):
                x_list, y = gen.get_slice(size=args.batch_size, dataframe=True, single=args.single)
                for j, input_feature in enumerate(x_list):
                    input_feature = input_feature.astype('float32')     # commonality 
                    store_key = 'x_{}_{}'.format(partition, j)          # HDFStore key, major 
                   
                    # cell_rnaseq names are stored in their own HDFS metadata file. 
                    # The saved names allow for feature set construction / gene selection by name  
                    # ??? 8-fold row repetition - why is this occurring                     ???????????????????????????????
                    
                    if j == 2:                                          # ugly ugly ugly ugly ugly ugly
                        if not loader.cell_feature_names:
                            loader.cell_feature_names = input_feature.columns[:].tolist()
                            colnames   = [cell_feature_names_container]
                            df_feature = pd.DataFrame(loader.cell_feature_names, columns=colnames)
                            pd_store   = pd.HDFStore(cell_feature_names_filename)
                            pd_store.put('feature_names', df_feature)
                            pd_store.close()

                        # The full cell descriptor is too wide for direct HDFS H5 representation, 
                        # Break it up into gulp_size chunks and write each independently.
                        # The column names are replaced with null-strings because (1) the
                        # header storage space is limitted and (2) they are not needed 

                        if True:
                            gulp_size   = 4096
                            _, nbr_cols = input_feature.shape
                            nbr_passes  = (nbr_cols + gulp_size - 1) // gulp_size
                            
                            for curr in range(nbr_passes): 
                                store_subkey = '{}_{}'.format(store_key, curr)      # HDFStore key, minor
                                colorg = gulp_size * curr
                                colfin = colorg + gulp_size    
                                subset = input_feature.iloc[:, colorg:colfin]
                                subset.columns = [''] * len(subset.columns)
                                store.append(store_subkey, subset, format='table')  # store sharded feature_set  
                            continue

                    # store standard feature set 
                    # minimize the contribution of unneeded column names to 64K HDFStore header size limit

                    input_feature.columns = [''] * len(input_feature.columns)
                    store.append(store_key, input_feature, format='table')

                # store labels
                store.append('y_{}'.format(partition), y.astype({target:'float32'}), format='table',
                        min_itemsize={'Sample':30, 'Drug1':10, 'Drug2':10})

                logger.info('Generating {} dataset. {} / {}'.format(partition, i, gen.steps))

                ### TEST ### TEST ### TEST ### TEST ### TEST ### TEST ### TEST ### TEST ### TEST 
                ### Read back and reassemble the feature sets written above. It is easier to test
                ### constructs here than in DataFeeder::__getitem__
              
               #"""
                if i == 0:
                    x = []
                    for j in range(7):
                        store_key = 'x_{}_{}'.format(partition, j)    
                        if j != 2:
                            t = store.select(store_key, start=0, stop=4096)
                            x.append(t)
                        else:
                            i = 0
                            df_accum = pd.DataFrame()
                            while True:
                                try:
                                    store_subkey = '{}_{}'.format(store_key, i)
                                    t = store.select(store_subkey, start=0, stop=4096)
                                except KeyError:
                                    break
                                df_accum = pd.concat([df_accum, t], axis=1)
                                i += 1
                            x.append(df_accum)
                    y = store.select('y_{}'.format(partition))['Growth'] 
               #"""
                ### TEST ### TEST ### TEST ### TEST ### TEST ### TEST ### TEST ### TEST ### TEST 
        # cleanup
        store.close()

        logger.info('Completed generating {}'.format(export_data_fname))
        return

    #----------------------------------------------------------------------------------------------    

    # Define additional loader attributes supporting dynamic feature-set content selection

    loader.cell_feature_names = None
    loader.feature_selections = None
    loader.feature_selection_xref = None
    
    # If training/evaluating with previously exported data, recover the names of the individual
    # features (genes) associated with the cell_rnaseq feature set from its metadata file 
    
    if args.use_exported_data:
        filename = args.use_exported_data + '.rnaseq_names'
        if not os.path.exists(filename):
            sys.exit("Feature name file: %s not found" % filename)

        pd_store  = pd.HDFStore(filename)
        names     = pd_store.get('feature_names')
        name_list = names.iloc[:,0].tolist()
        loader.cell_feature_names = {key:seq for (seq, key) in enumerate(name_list)}
        pd_store.close()

    # The 'cell_features_list' argument can be specified with 'use_exported_data' to define
    # a subset of features (genes) from the complete 'cell_rnaseq' feature set. It is optional,
    # however, and all features are selected in the absence of that arg.
    # The cell_features_list is case sensitive. 

    if args.cell_features_list:
        if not args.use_exported_data:
            sys.exit('Feature selection lists are only available with use_exported_data')

        featfile = args.cell_features_list
        if not os.path.exists(featfile):
            sys.exit("Cell feature selection file: %s not found" % featfile)
        
        with open(featfile) as f:
            text_list = f.readlines()
        
        loader.feature_selections = [line.strip() for line in text_list]
        loader.feature_selections = [line for line in loader.feature_selections if line != '']

        select_len = len(loader.feature_selections)
        if select_len == 0:
            sys.exit("The feature selection list: %s is empty" % featfile)

        # Duplicate feature names are allowed but most likely indicate a feature selection file error
        # Report all duplicate names (could be difficult to spot without programmed assistance)

        feature_set = set(loader.feature_selections)
        if len(feature_set) != select_len:
            print('The cell_features_list contains duplicate entries')
            report_duplicates(as_list=loader.feature_selections, as_set=feature_set, name=featfile)
    
        # replace raw 'cell.rnaseq' width with that of the 'selected features' list

        loader.feature_shapes['cell.rnaseq'] = (select_len,)

        # Construct feature_selection_xref list. Each entry contains an index into the full 
        # cell_rnaseq feature array. It is an error if a selected feature name cannot be matched
        # to the base feature set.

        berror = False
        loader.feature_selection_xref = []

        for selection in loader.feature_selections:
            featndx = loader.cell_feature_names.get(selection)
            if featndx:  
                loader.feature_selection_xref.append(featndx)
            else:
                berror = True
                logging.error('selected feature: %s not found' % selection)

        if berror:
            sys.exit('Terminating due to error')

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
    #
    #
    if args.cp:
        model_json = model.to_json()
        with open(prefix+'.model.json', 'w') as f:
            print(model_json, file=f)

    def warmup_scheduler(epoch):
        lr = args.learning_rate or base_lr * args.batch_size/100
        if epoch <= 5:
            K.set_value(model.optimizer.lr, (base_lr * (5-epoch) + lr * epoch) / 5)
        logger.debug('Epoch {}: lr={:.5g}'.format(epoch, K.get_value(model.optimizer.lr)))
        return K.get_value(model.optimizer.lr)

    df_pred_list = []

    cv_ext = ''
    cv = args.cv if args.cv > 1 else 1

    for fold in range(cv):
        if args.cv > 1:
            logger.info('Cross validation fold {}/{}:'.format(fold+1, cv))
            cv_ext = '.cv{}'.format(fold+1)

        if len(args.gpus) > 1:
            from keras.utils import multi_gpu_model
            gpu_count = len(args.gpus)
            model = multi_gpu_model(build_model(loader, args, silent=True), cpu_merge=False, gpus=gpu_count)
            max_queue_size = 10 * gpu_count
            use_multiprocessing = False
            workers = 1 # * gpu_count
        else:
            model = build_model(loader,args, silent=True)
            max_queue_size = 10
            use_multiprocessing = False
            workers = 1


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
        checkpointer = ModelCheckpoint(prefix+cv_ext+'.weights.h5', save_best_only=True, save_weights_only=True)
        tensorboard = TensorBoard(log_dir="tb/tb{}{}".format(ext, cv_ext))
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

        if args.no_gen:
            x_train_list, y_train = train_gen.get_slice(size=train_gen.size, single=args.single)
            x_val_list, y_val = val_gen.get_slice(size=val_gen.size, single=args.single)
            history = model.fit(x_train_list, y_train,
                                batch_size=args.batch_size,
                                epochs=args.epochs,
                                callbacks=callbacks,
                                validation_data=(x_val_list, y_val))
        else:
            logger.info('Data points per epoch: train = %d, val = %d',train_gen.size, val_gen.size)
            logger.info('Steps per epoch: train = %d, val = %d',train_gen.steps, val_gen.steps)
            history = model.fit_generator(train_gen, train_gen.steps,
                                          epochs=args.epochs,
                                          callbacks=callbacks,
                                          max_queue_size=max_queue_size,
                                          use_multiprocessing=use_multiprocessing,
                                          workers=workers,
                                          validation_data=val_gen,
                                          validation_steps=val_gen.steps)

        if args.cp:
            model = model_recorder.best_model
            model.save(prefix+'.model.h5')
            # model.load_weights(prefix+cv_ext+'.weights.h5')

        if args.no_gen:
            y_val_pred = model.predict(x_val_list, batch_size=args.batch_size)
        else:
            val_gen.reset()
            y_val_pred = model.predict_generator(val_gen, val_gen.steps+1)
            y_val_pred = y_val_pred[:val_gen.size]

        y_val_pred = y_val_pred.flatten()

        scores = evaluate_prediction(y_val, y_val_pred)
        log_evaluation(scores)

        df_val = df_val.assign(PredictedGrowth=y_val_pred, GrowthError=y_val_pred-y_val)
        df_val['Predicted'+target] = y_val_pred
        df_val[target+'Error'] = y_val_pred-y_val
        df_pred_list.append(df_val)

        plot_history(prefix, history, 'loss')
        plot_history(prefix, history, 'r2')

    pred_fname = prefix + '.predicted.tsv'
    df_pred = pd.concat(df_pred_list)
    if args.agg_dose:
        df_pred.sort_values(['Source', 'Sample', 'Drug1', 'Drug2', target], inplace=True)
    else:
        df_pred.sort_values(['Sample', 'Drug1', 'Drug2', 'Dose1', 'Dose2', 'Growth'], inplace=True)
    df_pred.to_csv(pred_fname, sep='\t', index=False, float_format='%.4g')

    if args.cv > 1:
        scores = evaluate_prediction(df_pred[target], df_pred['Predicted'+target])
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
