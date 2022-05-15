from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_core.enitiers.split_params import SplittingParams


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def split_train_val_data(
    data: pd.DataFrame, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    return train_data, val_data


def remove_outliers(df):
    df = df.drop(df[(df['chol'] > 350)].index)
    df = df.drop(df[(df['thalach'] < 80)].index)
    df = df.drop(df[(df['oldpeak'] > 4)].index)
    return df
