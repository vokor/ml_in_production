#!/usr/bin/env python3
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import logging
import logging.config

import yaml

from ml_core.data.make_dataset import read_data, split_train_val_data, remove_outliers
from ml_core.enitiers.train_pipeline_params import read_training_pipeline_params
from ml_core.features.build_features import extract_target, build_transformer, make_features
from ml_core.models.model_fit_predict import train_model, create_inference_pipeline, predict_model, evaluate_model, \
    serialize_model, deserialize_model

logger = logging.getLogger("ml_classifier")


def setup_logging(logging_yaml_config_fpath):
    """setup logging via YAML if it is provided"""
    if logging_yaml_config_fpath:
        with open(logging_yaml_config_fpath) as config_fin:
            logging.config.dictConfig(yaml.safe_load(config_fin))


def callback_train(arguments):
    setup_logging(arguments.logging_yaml_config_fpath)
    training_pipeline_params = read_training_pipeline_params(arguments.train_config)
    inference_pipeline = process_train(training_pipeline_params)
    path_to_model = serialize_model(
        inference_pipeline, training_pipeline_params.output_model_path
    )
    return path_to_model


def process_train(training_pipeline_params):
    logger.debug(f"start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    data = remove_outliers(data)
    logger.debug(f"data.shape is {data.shape}")

    train_df, val_df = split_train_val_data(
        data, training_pipeline_params.splitting_params
    )

    train_target = extract_target(train_df, training_pipeline_params.feature_params)
    val_target = extract_target(val_df, training_pipeline_params.feature_params)
    train_df = train_df.drop(training_pipeline_params.feature_params.target_col, 1)
    val_df = val_df.drop(training_pipeline_params.feature_params.target_col, 1)

    logger.debug(f"train_df.shape is {train_df.shape}")
    logger.debug(f"val_df.shape is {val_df.shape}")
    transformer = build_transformer(training_pipeline_params.feature_params)
    transformer.fit(train_df)
    train_features = make_features(transformer, train_df)
    logger.debug(f"train_features.shape is {train_features.shape}")
    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )
    inference_pipeline = create_inference_pipeline(model, transformer)

    predicts = predict_model(inference_pipeline, val_df)
    metrics = evaluate_model(predicts, val_target)

    save_metrics(metrics, training_pipeline_params.metric_path)

    return inference_pipeline


def save_metrics(metrics, path):
    with open(path, "w") as metric_file:
        json.dump(metrics, metric_file)
    logger.debug(f"metrics is {metrics}")


def callback_eval(arguments):
    inference_pipeline = deserialize_model(arguments.inference_pipeline)
    data = read_data(arguments.data)
    predicts = predict_model(inference_pipeline, data)
    np.savetxt(arguments.output, predicts, delimiter=",", fmt='%i')


def setup_parser(parser):
    subparsers = parser.add_subparsers(help="choose command")

    train_parser = subparsers.add_parser(
        "train",
        help="train model according to train_config.yaml",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument(
        "--train-config", dest="train_config",
        default=None, help="path to logging config in YAML format", required=True
    )
    train_parser.add_argument(
        "--logging-config", dest="logging_yaml_config_fpath",
        default=None, help="path to logging config in YAML format",
    )
    train_parser.set_defaults(callback=callback_train)

    eval_parser = subparsers.add_parser(
        "eval",
        help="evaluate model",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    eval_parser.add_argument(
        "-i", "--inference_pipeline",
        required=True,
        help="path to serialized inference pipeline",
    )
    eval_parser.add_argument(
        "-d", "--data",
        required=True,
        help="path to data need to be predicted",
    )
    eval_parser.add_argument(
        "-o", "--output",
        required=True,
        help="path to store result",
    )
    eval_parser.set_defaults(callback=callback_eval)

    return parser


def main():
    parser = ArgumentParser(
        prog="ml pipeline classifier",
        description="tool to classify Heart Disease",
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)


if __name__ == "__main__":
    main()
