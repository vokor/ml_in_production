import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ml_core.enitiers.train_params import TrainingParams
from ml_core.models.naive_bayes import NaiveBayes
from sklearn.metrics import accuracy_score

ClassificationModel = Union[NaiveBayes]


def train_model(
        features: pd.DataFrame, target: pd.Series, train_params: TrainingParams
) -> ClassificationModel:
    if train_params.model_type == "NaiveBayes":
        model = NaiveBayes()
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(
        model: Pipeline, features: pd.DataFrame
) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(
        predicts: np.ndarray, target: pd.Series
) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(target, predicts)
    }


def create_inference_pipeline(
        model: ClassificationModel, transformer: ColumnTransformer
) -> Pipeline:
    return Pipeline([("feature_part", transformer), ("model_part", model)])


def serialize_model(model: object, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output


def deserialize_model(path: str) -> Pipeline:
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model
