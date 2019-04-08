#! /usr/bin/env python

import argparse
import logging

import keras
import pandas as pd

from uno_data import DataFeeder
from uno_baseline_keras2 import evaluate_prediction, log_evaluation

def log_evaluation(metric_outputs, description='Comparing y_true and y_pred:'):
    print(description)
    for metric, value in metric_outputs.items():
        print('  {}: {:.4f}'.format(metric, value))


def get_parser():
    parser = argparse.ArgumentParser(description='Uno infer script')
    parser.add_argument("--data",
                       help="data file to infer on. expect exported file from uno_baseline_keras2.py")
    parser.add_argument("--model_file", help="json model description file")
    parser.add_argument("--weights_file", help="model weights file")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    test_gen = DataFeeder(filename=args.data, batch_size=1024)

    if 'json' == args.model_file.split('.')[-1]:
        with open(args.model_file, 'r') as f:
            model_json = f.read()
            model = keras.models.model_from_json(model_json)
            model.load_weights(args.weights_file, by_name=True)
    else:
        model = keras.models.load_model(args.model_file, compile=False)
        model.load_weights(args.weights_file)

    model.summary()

    cv_pred_list = []
    cv_y_list = []
    df_pred_list = []
    cv_stats = {'mae': [], 'mse': [], 'r2': [], 'corr': []}
    for cv in range(args.n_pred):
        cv_pred = []
        dataset = ['train', 'val'] if args.partition == 'all' else [args.partition]
        for partition in dataset:
            test_gen = DataFeeder(filename=args.data, partition=partition, batch_size=1024)
            y_test_pred = model.predict_generator(test_gen, test_gen.steps)
            y_test_pred = y_test_pred.flatten()

            df_y = test_gen.get_response(copy=True)
            y_test = df_y['Growth'].values

            df_pred = df_y.assign(PredictedGrowth=y_test_pred, GrowthError=y_test_pred - y_test)
            df_pred_list.append(df_pred)
            test_gen.close()

            if cv == 0:
                cv_y_list.append(df_y)
            cv_pred.append(y_test_pred)
        cv_pred_list.append(np.concatenate(cv_pred))

        # calcuate stats for mse, mae, r2, corr
        scores = evaluate_prediction(df_pred['Growth'], df_pred['PredictedGrowth'])
        # log_evaluation(scores, description=cv)
        [cv_stats[key].append(scores[key]) for key in scores.keys()]

    df_pred = pd.concat(df_pred_list)
    cv_y = pd.concat(cv_y_list)

    # save to tsv
    df_pred = df_y.assign(PredictedGrowth=y_test_pred, GrowthError=y_test_pred-y_test)
    df_pred.sort_values(['Sample', 'Drug1', 'Drug2', 'Dose1', 'Dose2', 'Growth'], inplace=True)
    df_pred.to_csv('prediction.tsv', sep='\t', index=False, float_format='%.4g')


if __name__ == '__main__':
    main()

