from typing import Optional

from dataclasses import dataclass

import yaml

from ml_core.enitiers.feature_params import FeatureParams
from ml_core.enitiers.split_params import SplittingParams
from ml_core.enitiers.train_params import TrainingParams

from marshmallow_dataclass import class_schema


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
